import os
import numpy as np
import torch
from PIL import Image
import triton_python_backend_utils as pb_utils
from diffusers import DPMSolverMultistepScheduler

from .src.stable_diffusion_allinone_pipeline import StableDiffusionAllInOnePipeline
from .src.utils import MODE_LIST, DATA_TO_CAPTION
from .src.processes import process_t2i, process_i2i, process_inpaint, process_outpaint, process_add


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
        # self.pipe.set_progress_bar_config(disable=True)
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
            data_name = pb_utils.get_input_tensor_by_name(request, "data_name").as_numpy()[0].decode()
            mode = pb_utils.get_input_tensor_by_name(request, "mode").as_numpy()[0].decode()
            class_name_to_inpaint = pb_utils.get_input_tensor_by_name(request, "class_name_to_inpaint").as_numpy()[0].decode()
            raw_image = pb_utils.get_input_tensor_by_name(request, "raw_image").as_numpy()
            mask_image = pb_utils.get_input_tensor_by_name(request, "mask_image").as_numpy()
            boxes = pb_utils.get_input_tensor_by_name(request, "boxes").as_numpy()
            phrases = pb_utils.get_input_tensor_by_name(request, "phrases").as_numpy()
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0].decode()
            negative_prompt = pb_utils.get_input_tensor_by_name(request, "negative_prompt").as_numpy()[0].decode()
            num_inference_steps = pb_utils.get_input_tensor_by_name(request, "num_inference_steps").as_numpy()[0]
            guidance_scale = pb_utils.get_input_tensor_by_name(request, "guidance_scale").as_numpy()[0]
            strength = pb_utils.get_input_tensor_by_name(request, "strength").as_numpy()[0]
            strength_2 = pb_utils.get_input_tensor_by_name(request, "strength_2").as_numpy()[0]
            num_images_per_prompt = pb_utils.get_input_tensor_by_name(request, "num_images_per_prompt").as_numpy()[0]
            seed = pb_utils.get_input_tensor_by_name(request, "seed").as_numpy()[0]
            
            raw_image = Image.fromarray(raw_image)
            mask_image = Image.fromarray(mask_image)
            phrases = [x.decode() for x in phrases]
            seed = int(seed)

            # check mode
            if mode not in MODE_LIST:
                assert False, f"not supproted mode: {mode}, supported: {MODE_LIST}"

            # check data_name
            if data_name not in DATA_TO_CAPTION:
                assert False, f"not supported data name: {data_name}, supported: {list(DATA_TO_CAPTION.keys())}"

            # Get base prompt
            prompt = DATA_TO_CAPTION[data_name](None) + f" {prompt}"

            # Update LoRA layer
            if self.cur_lora_name is None or self.cur_lora_name != data_name:
                self.update_lora(data_name)

            kwargs = {
                "pipe": self.pipe,
                "data_name": data_name,
                "mode": mode,
                "class_name_to_inpaint": class_name_to_inpaint,
                "raw_image": raw_image,
                "mask_image": mask_image,
                "boxes": boxes,
                "phrases": phrases,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "strength": strength,
                "strength_2": strength_2,
                "num_images_per_prompt": num_images_per_prompt,
                "seed": seed,
                "width": 512,
                "height": 512,
            }

            # inference
            if mode == "Text-to-Image":
                results = process_t2i(**kwargs)
            elif mode == "Image-to-Image":
                results = process_i2i(**kwargs)
            elif mode == "Inpaint":
                results = process_inpaint(**kwargs)
            elif mode == "Outpaint":
                results = process_outpaint(**kwargs)
            elif mode == "Add":
                results = process_add(**kwargs)
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
