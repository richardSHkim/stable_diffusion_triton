import os
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from diffusers import DPMSolverMultistepScheduler

from .src.stable_diffusion_allinone_pipeline import StableDiffusionAllInOnePipeline
from .src.utils import create_reco_prompt, DATA_TO_CAPTION


class TritonPythonModel:
    def initialize(self, args):
        self.device = f"cuda:{args['model_instance_device_id']}"
        self.model_dir = os.path.join(args["model_repository"], args["model_version"])

        self.pipe = StableDiffusionAllInOnePipeline.from_pretrained(
            "j-min/reco_sd14_laion",
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe = self.pipe.to(self.device)

        self.cur_lora_name = None

    def update_lora(self, data_name):
        self.pipe.unload_lora_weights()
        self.pipe.load_lora_weights(
            os.path.join(self.model_dir, "weights", data_name),
            "pytorch_lora_weights.safetensors",
        )
        self.cur_lora_name = data_name
        print(f"Loaded {data_name} LoRA weights")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get inputs
            data_name = pb_utils.get_input_tensor_by_name(request, "data_name").as_numpy()[0][0].decode()
            mode = pb_utils.get_input_tensor_by_name(request, "mode").as_numpy()[0][0].decode()
            class_name_to_inpaint = pb_utils.get_input_tensor_by_name(request, "class_name_to_inpaint").as_numpy()[0][0].decode()
            raw_image = pb_utils.get_input_tensor_by_name(request, "raw_image").as_numpy()
            mask_image = pb_utils.get_input_tensor_by_name(request, "mask_image").as_numpy()
            boxes = pb_utils.get_input_tensor_by_name(request, "boxes").as_numpy()
            phrases = pb_utils.get_input_tensor_by_name(request, "phrases").as_numpy()
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()
            negative_prompt = pb_utils.get_input_tensor_by_name(request, "negative_prompt").as_numpy()
            num_inference_steps = pb_utils.get_input_tensor_by_name(request, "num_inference_steps").as_numpy()[0][0]
            guidance_scale = pb_utils.get_input_tensor_by_name(request, "guidance_scale").as_numpy()[0][0]
            strength = pb_utils.get_input_tensor_by_name(request, "strength").as_numpy()[0][0]
            strength_2 = pb_utils.get_input_tensor_by_name(request, "strength_2").as_numpy()[0][0]
            num_images_per_prompt = pb_utils.get_input_tensor_by_name(request, "num_images_per_prompt").as_numpy()[0][0]
            seed = pb_utils.get_input_tensor_by_name(request, "seed").as_numpy()[0][0]
            
            phrases = [[x.decode() for x in phrases_] for phrases_ in phrases]
            prompt = [x[0].decode() for x in prompt]
            negative_prompt = [x[0].decode() for x in negative_prompt]
            seed = int(seed)
            
            width = 512
            height = 512
            generator=torch.Generator(device=self.device).manual_seed(seed) if seed >=0 else None

            # check mode
            if mode not in ["Text-to-Image", "Image-to-Image", "Inpaint", "Outpaint", "Add"]:
                continue

            # check data_name
            if data_name not in DATA_TO_CAPTION:
                continue

            # Get base prompt
            prompt = [DATA_TO_CAPTION[data_name](None) + f" {x}" for x in prompt]

            # Get ReCo prompt
            reco_prompt = []
            for prompt_, phrases_, boxes_ in zip(prompt, phrases, boxes):
                reco_prompt_ = create_reco_prompt(caption=prompt_, 
                                            phrases=phrases_, 
                                            boxes=boxes_, 
                                            normalize_boxes=True, 
                                            w=width, 
                                            h=height,
                                            )
                reco_prompt.append(reco_prompt_)
            prompt = reco_prompt

            # Update LoRA layer
            if self.cur_lora_name is None or self.cur_lora_name != data_name:
                self.update_lora(data_name)

            # inference
            if mode == "Text-to-Image":
                results = self.pipe(prompt,
                                    negative_prompt=negative_prompt,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    num_images_per_prompt=num_images_per_prompt,
                                    generator=generator,
                                    output_type="np",
                                    ).images
            else:
                raise NotImplementedError

            # Response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_image",
                        results.astype(np.float32),
                    )
                ]
            )
            responses.append(inference_response)
        return responses