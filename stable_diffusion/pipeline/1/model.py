import json
from typing import Any, Callable, Dict, List, Optional, Union
import os
from tqdm.auto import tqdm

import inspect
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from diffusers import DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer


class TritonPythonModel:
    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "generated_image"
            )["data_type"]
        )
        model_dir = os.path.join(args["model_repository"], args["model_version"])
        self.tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_dir, subfolder="scheduler", 
                                                                     use_karras_sigmas=True)
        self.vae_config = json.load(open(os.path.join(model_dir, "vae_config.json"), "r"))
        self.vae_scale_factor = 2 ** (len(self.vae_config["block_out_channels"]) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get inputs
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0][0].decode()
            negative_prompt = pb_utils.get_input_tensor_by_name(request, "negative_prompt").as_numpy()[0][0].decode()
            guidance_scale = pb_utils.get_input_tensor_by_name(request, "guidance_scale").as_numpy()[0][0]
            num_inference_steps = pb_utils.get_input_tensor_by_name(request, "num_inference_steps").as_numpy()[0][0]
            seed = int(pb_utils.get_input_tensor_by_name(request, "seed").as_numpy()[0][0])

            batch_size = 1
            num_images_per_prompt = 1
            height = 512
            width = 512
            device = torch.device("cuda")
            eta = None
            self.cross_attention_kwargs = None
            self.do_classifier_free_guidance = (guidance_scale > 1.0)
            self.guidance_rescale = 0.0
            generator = torch.Generator(device=device).manual_seed(seed)

            # 1. Check inputs. Raise error if not correct
            # 2. Define call parameters

            # 3. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device)

            # 5. Prepare latent variables
            num_channels_latents = 4
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.call_unet(
                        latent_model_input,
                        t,
                        prompt_embeds
                    )

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    # update probress bar
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
            
            image = self.call_vae(latents)
            image = self.image_processor.postprocess(image, output_type="np", do_denormalize=[True] * image.shape[0])

            # Sending results
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_image",
                        np.array(image, dtype=self.output_dtype),
                    )
                ]
            )
            responses.append(inference_response)
        return responses

    def call_text_encoder(self, text_input_ids) -> torch.Tensor:
        input_ids = text_input_ids.type(dtype=torch.int32)
        inputs = [
            pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))
        ]
        request = pb_utils.InferenceRequest(
            model_name="text_encoder",
            requested_output_names=["text_embeddings"],
            inputs=inputs,
        )
        response = request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(
                response.error().message()
            )
        else:
            prompt_embeds = pb_utils.get_output_tensor_by_name(
                response, "text_embeddings"
            )
            prompt_embeds = torch.from_dlpack(prompt_embeds.to_dlpack())
        return prompt_embeds

    def call_unet(self, latent_model_input, t, prompt_embeds) -> torch.Tensor:
        sample = pb_utils.Tensor.from_dlpack("sample", 
                                             torch.to_dlpack(latent_model_input))
        timestep = pb_utils.Tensor.from_dlpack("timestep", 
                                               torch.to_dlpack(t[None].to(dtype=torch.int32)))
        encoder_hidden_states = pb_utils.Tensor.from_dlpack("encoder_hidden_states", 
                                                            torch.to_dlpack(prompt_embeds))
        request = pb_utils.InferenceRequest(
            model_name="unet",
            requested_output_names=["latent"],
            inputs=[sample, timestep, encoder_hidden_states],
        )

        response = request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        else:
            noise_pred = pb_utils.get_output_tensor_by_name(
                response, "latent"
            )
            noise_pred = torch.from_dlpack(noise_pred.to_dlpack())
        return noise_pred

    def call_vae(self, latents) -> torch.Tensor:
        latents = latents / self.vae_config["scaling_factor"]
        input_latent = pb_utils.Tensor.from_dlpack(
            "latent", torch.to_dlpack(latents.to(dtype=torch.float32))
        )
        request = pb_utils.InferenceRequest(
            model_name="vae",
            requested_output_names=["images"],
            inputs=[input_latent],
        )

        response = request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        else:
            decoded_image = pb_utils.get_output_tensor_by_name(
                response, "images"
            )
            decoded_image = torch.from_dlpack(decoded_image.to_dlpack())
        return decoded_image

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.call_text_encoder(text_input_ids)

        prompt_embeds_dtype = torch.float16
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = self.call_text_encoder(uncond_input.input_ids)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")