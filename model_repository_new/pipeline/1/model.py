import inspect
import json
import os
from typing import List, Optional

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    retrieve_latents,
    retrieve_timesteps,
)
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm.auto import tqdm
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
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_dir, subfolder="scheduler", use_karras_sigmas=True
        )
        self.vae = AutoencoderKL.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=torch.float16
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config["block_out_channels"]) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def execute(self, requests):
        self.responses = []
        for request in requests:
            # Get inputs
            self.mode = (
                pb_utils.get_input_tensor_by_name(request, "mode")
                .as_numpy()[0]
                .decode()
            )
            self.prompt = (
                pb_utils.get_input_tensor_by_name(request, "prompt")
                .as_numpy()[0]
                .decode()
            )
            self.negative_prompt = (
                pb_utils.get_input_tensor_by_name(request, "negative_prompt")
                .as_numpy()[0]
                .decode()
            )
            self.image = Image.fromarray(
                pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            )
            self.mask_image = Image.fromarray(
                pb_utils.get_input_tensor_by_name(request, "mask_image").as_numpy()
            )
            self.guidance_scale = pb_utils.get_input_tensor_by_name(
                request, "guidance_scale"
            ).as_numpy()[0]
            self.num_inference_steps = pb_utils.get_input_tensor_by_name(
                request, "num_inference_steps"
            ).as_numpy()[0]
            self.strength = pb_utils.get_input_tensor_by_name(
                request, "strength"
            ).as_numpy()[0]
            self.num_images_per_prompt = pb_utils.get_input_tensor_by_name(
                request, "num_images_per_prompt"
            ).as_numpy()[0]
            self.seed = int(
                pb_utils.get_input_tensor_by_name(request, "seed").as_numpy()[0]
            )

            self.batch_size = 1
            self.height = 512
            self.width = 512
            self.device = torch.device("cuda")
            self.eta = None
            self.cross_attention_kwargs = None
            self.do_classifier_free_guidance = self.guidance_scale > 1.0
            self.guidance_rescale = 0.0
            self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
            self.vae = self.vae.to(self.device)

            if self.mode == "t2i":
                self.run_t2i()
            elif self.mode == "i2i":
                self.run_i2i()
            elif self.mode == "inpaint":
                self.run_inpaint()
            else:
                raise NotImplementedError

        return self.responses

    def run_t2i(self):
        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            self.prompt,
            self.device,
            self.num_images_per_prompt,
            self.do_classifier_free_guidance,
            self.negative_prompt,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, self.num_inference_steps, self.device
        )

        # 5. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            self.batch_size * self.num_images_per_prompt,
            num_channels_latents,
            self.height,
            self.width,
            prompt_embeds.dtype,
            self.device,
            self.generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, self.eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.call_unet(latent_model_input, t, prompt_embeds)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # update probress bar
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        image = self.call_vae(latents)
        image = self.image_processor.postprocess(
            image, output_type="np", do_denormalize=[True] * image.shape[0]
        )

        # Sending results
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    np.array(image, dtype=self.output_dtype),
                )
            ]
        )
        self.responses.append(inference_response)
        return

    def run_i2i(self):
        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            self.prompt,
            self.device,
            self.num_images_per_prompt,
            self.do_classifier_free_guidance,
            self.negative_prompt,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Preprocess image
        image = self.image_processor.preprocess(self.image)

        # 5. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, self.num_inference_steps, self.device, None
        )
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, self.strength, self.device
        )
        latent_timestep = timesteps[:1].repeat(
            self.batch_size * self.num_images_per_prompt
        )

        # 6. Prepare latent variables
        latents = self.prepare_latents_i2i(
            image,
            latent_timestep,
            self.batch_size,
            self.num_images_per_prompt,
            prompt_embeds.dtype,
            self.device,
            self.generator,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, self.eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.call_unet(latent_model_input, t, prompt_embeds)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        image = self.call_vae(latents)
        image = self.image_processor.postprocess(
            image, output_type="np", do_denormalize=[True] * image.shape[0]
        )
        # Sending results
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    np.array(image, dtype=self.output_dtype),
                )
            ]
        )
        self.responses.append(inference_response)

    def run_inpaint(self, strength_2=0):
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        if strength_2 == 0:
            strength_2 = None

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            self.prompt,
            self.device,
            self.num_images_per_prompt,
            self.do_classifier_free_guidance,
            self.negative_prompt,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, self.num_inference_steps, self.device, None
        )
        if strength_2 is not None:
            _, num_inference_steps_2 = self.get_timesteps(
                num_inference_steps=num_inference_steps,
                strength=strength_2,
                device=self.device,
            )
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps,
            strength=self.strength,
            device=self.device,
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {self.strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(
            self.batch_size * self.num_images_per_prompt
        )
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = self.strength == 1.0

        # 5. Preprocess mask and image
        padding_mask_crop = None
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(
                self.mask_image, self.width, self.height, pad=padding_mask_crop
            )
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        # original_image = self.image
        init_image = self.image_processor.preprocess(
            self.image,
            height=self.height,
            width=self.width,
            crops_coords=crops_coords,
            resize_mode=resize_mode,
        )
        init_image = init_image.to(dtype=torch.float32)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = 4
        return_image_latents = num_channels_unet == 4

        latents_outputs = self.prepare_latents_inpaint(
            self.batch_size * self.num_images_per_prompt,  # type: ignore
            num_channels_latents,
            self.height,
            self.width,
            prompt_embeds.dtype,
            self.device,
            self.generator,
            latents=None,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            self.mask_image,
            height=self.height,
            width=self.width,
            resize_mode=resize_mode,
            crops_coords=crops_coords,
        )

        masked_image_latents = None
        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            self.batch_size * self.num_images_per_prompt,
            self.height,
            self.width,
            prompt_embeds.dtype,
            self.device,
            self.generator,
            self.do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if (
                num_channels_latents + num_channels_mask + num_channels_masked_image
                != self.unet.config.in_channels
            ):
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, self.eta)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if num_channels_unet == 9:
                    latent_model_input = torch.cat(
                        [latent_model_input, mask, masked_image_latents], dim=1
                    )

                # predict the noise residual
                noise_pred = self.call_unet(latent_model_input, t, prompt_embeds)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                if num_channels_unet == 4:
                    if (strength_2 is None) or (
                        i <= len(timesteps) - num_inference_steps_2
                    ):
                        init_latents_proper = image_latents
                        if self.do_classifier_free_guidance:
                            init_mask, _ = mask.chunk(2)
                        else:
                            init_mask = mask

                        if i < len(timesteps) - 1:
                            noise_timestep = timesteps[i + 1]
                            init_latents_proper = self.scheduler.add_noise(
                                init_latents_proper,
                                noise,
                                torch.tensor([noise_timestep]),
                            )

                        latents = (
                            1 - init_mask
                        ) * init_latents_proper + init_mask * latents
                if (strength_2 is not None) and (
                    i == len(timesteps) - num_inference_steps_2 + 1
                ):
                    print(f"skip pasting from step {i}!")

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        image = self.call_vae(latents)
        image = self.image_processor.postprocess(
            image, output_type="np", do_denormalize=[True] * image.shape[0]
        )

        # Sending results
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    np.array(image, dtype=self.output_dtype),
                )
            ]
        )
        self.responses.append(inference_response)
        return

    def call_text_encoder(self, text_input_ids) -> torch.Tensor:
        input_ids = text_input_ids.type(dtype=torch.int32)
        inputs = [pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))]
        request = pb_utils.InferenceRequest(
            model_name="text_encoder",
            requested_output_names=["text_embeddings"],
            inputs=inputs,
        )
        response = request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        else:
            prompt_embeds = pb_utils.get_output_tensor_by_name(
                response, "text_embeddings"
            )
            prompt_embeds = torch.from_dlpack(prompt_embeds.to_dlpack())
        return prompt_embeds

    def call_unet(self, latent_model_input, t, prompt_embeds) -> torch.Tensor:
        sample = pb_utils.Tensor.from_dlpack(
            "sample", torch.to_dlpack(latent_model_input)
        )
        timestep = pb_utils.Tensor.from_dlpack(
            "timestep", torch.to_dlpack(t[None].to(dtype=torch.int32))
        )
        encoder_hidden_states = pb_utils.Tensor.from_dlpack(
            "encoder_hidden_states", torch.to_dlpack(prompt_embeds)
        )
        request = pb_utils.InferenceRequest(
            model_name="unet",
            requested_output_names=["latent"],
            inputs=[sample, timestep, encoder_hidden_states],
        )

        response = request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        else:
            noise_pred = pb_utils.get_output_tensor_by_name(response, "latent")
            noise_pred = torch.from_dlpack(noise_pred.to_dlpack())
        return noise_pred

    def call_vae(self, latents) -> torch.Tensor:
        latents = latents / self.vae.config["scaling_factor"]
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
            decoded_image = pb_utils.get_output_tensor_by_name(response, "images")
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
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
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
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

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

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_latents_i2i(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
    ):
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(
                        self.vae.encode(image[i : i + 1]), generator=generator[i]
                    )
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(
                    self.vae.encode(image), generator=generator
                )

            init_latents = self.vae.config.scaling_factor * init_latents

        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate(
                "len(prompt) != len(image)",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def prepare_latents_inpaint(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                elif isinstance(generator, list):
                    image_latents = [
                        retrieve_latents(
                            self.vae.encode(image[i : i + 1]), generator=generator[i]
                        )
                        for i in range(batch_size)
                    ]
                    image_latents = torch.cat(image_latents, dim=0)
                else:
                    image_latents = retrieve_latents(
                        self.vae.encode(image), generator=generator
                    )

                image_latents = self.vae.config.scaling_factor * image_latents
            image_latents = image_latents.repeat(
                batch_size // image_latents.shape[0], 1, 1, 1
            )

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = (
                noise
                if is_strength_max
                else self.scheduler.add_noise(image_latents, noise, timestep)
            )
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = (
                latents * self.scheduler.init_noise_sigma
                if is_strength_max
                else latents
            )
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                masked_image_latents = [
                    retrieve_latents(
                        self.vae.encode(masked_image[i : i + 1]), generator=generator[i]
                    )
                    for i in range(batch_size)
                ]
                masked_image_latents = torch.cat(masked_image_latents, dim=0)
            else:
                masked_image_latents = retrieve_latents(
                    self.vae.encode(masked_image), generator=generator
                )

            masked_image_latents = self.vae.config.scaling_factor * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

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
