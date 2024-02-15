#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import argparse
import json
import gc

import torch
from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.loaders import LoraLoaderMixin


class BaseModel():
    def __init__(
        self,
        fp16=False,
        path="",
        embedding_dim=768,
        text_maxlen=77,
        vae_scale_factor=8,
        min_batch=1,
        max_batch=4,
        min_image_shape=256,
        max_image_shape=1024,
        device="cuda",
        name="SD Model",
    ):
        self.fp16 = fp16
        self.path = path
        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen
        self.vae_scale_factor = vae_scale_factor
        self.device = device
        self.name = name

        self.min_batch = min_batch
        self.max_batch = max_batch
        self.min_image_shape = min_image_shape
        self.max_image_shape = max_image_shape
        self.min_latent_shape = self.min_image_shape // self.vae_scale_factor
        self.max_latent_shape = self.max_image_shape // self.vae_scale_factor

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_shape(self, batch_size, image_height, image_width):
        pass

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % self.vae_scale_factor == 0 or image_width % self.vae_scale_factor == 0
        latent_height = image_height // self.vae_scale_factor
        latent_width = image_width // self.vae_scale_factor
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)


class TextEncoder(BaseModel):
    def __init__(self, **kwargs):
        super(TextEncoder, self).__init__(**kwargs)

    def get_model(self, data_name):
        text_encoder = CLIPTextModel.from_pretrained(
            self.path,
            subfolder="text_encoder",
        )

        # load lora
        if os.path.isfile(os.path.join("weights", data_name, "pytorch_lora_weights.safetensors")):
            state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
                os.path.join("weights", data_name), 
                weight_name="pytorch_lora_weights.safetensors",
            )
            LoraLoaderMixin.load_lora_into_text_encoder(
                state_dict,
                network_alphas,
                text_encoder,
            )
        else:
            assert False, "lora file is not exist."
        
        text_encoder = text_encoder.to(self.device)
        return text_encoder

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
       return ['text_embeddings', 'pooler_output']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def get_input_shape(self, batch_size, image_height, image_width):
        return f"input_ids:{batch_size}x{self.text_maxlen}"


class UNet(BaseModel):
    def __init__(self, unet_dim=4, **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.unet_dim = unet_dim

    def get_model(self, data_name):
        unet = UNet2DConditionModel.from_pretrained(
            self.path,
            subfolder="unet",
        )

        # load lora
        if os.path.isfile(os.path.join("weights", data_name, "pytorch_lora_weights.safetensors")):
            state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
                os.path.join("weights", data_name), 
                weight_name="pytorch_lora_weights.safetensors",
            )
            LoraLoaderMixin.load_lora_into_unet(
                state_dict,
                network_alphas,
                unet,
            )
        else:
            assert False, "lora file is not exist."
        
        unet = unet.to(self.device, dtype=torch.float16 if self.fp16 else torch.float32)
        return unet

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '2B'},
            'latent': {0: '2B', 2: 'H', 3: 'W'}
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(2*batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device),
            torch.tensor([1.], dtype=torch.int32, device=self.device),
            torch.randn(2*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device)
        )
    
    def get_input_shape(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return f"sample:{batch_size*2}x{self.unet_dim}x{latent_height}x{latent_width},encoder_hidden_states:{batch_size*2}x{self.text_maxlen}x{self.embedding_dim}"


class VAE(BaseModel):
    def __init__(self, unet_dim=4, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.unet_dim = unet_dim

    def get_model(self, data_name):
        vae = AutoencoderKL.from_pretrained(
            self.path, 
            subfolder="vae",
        ).to(self.device)
        vae.forward = vae.decode
        return vae

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device)

    def get_input_shape(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return f"latent:{batch_size}x{self.unet_dim}x{latent_height}x{latent_width}"


def main(args):
    # configs
    min_batch_size = 1
    min_image_height = 512
    min_image_width = 512

    opt_batch_size = 4
    opt_image_height = 512
    opt_image_width = 512

    max_batch_size = 4
    max_image_height = 512
    max_image_width = 512

    # dirs
    model_dir = "weights/reco"
    # base_path = "deploy/triton_convert/docker/stable_diffusion"
    # model_repository = "model_repository"
    # model_version = "1"

    # Prepare: download pretrained weights and create `model_repository` directory.
    commands = [
        # f"cp -r {base_path} {model_repository}",
        # f"cp -r {model_dir}/scheduler/* {model_repository}/pipeline/{model_version}/scheduler/",
        # f"cp -r {model_dir}/tokenizer/* {model_repository}/pipeline/{model_version}/tokenizer/",
        # f"cp {model_dir}/vae/config.json {model_repository}/pipeline/{model_version}/vae_config.json",
        # f"mv {model_dir}/engines/vae.plan {model_repository}/vae/{model_version}/model.plan",
        f"mkdir -p onnx/{args.data_name}",
        f"mkdir -p engine/{args.data_name}",
    ]
    for cmd in commands:
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise Exception("Exit code", exit_code)

    # Prepare: get model instances
    text_encoder_config = json.load(open(os.path.join(model_dir, "text_encoder/config.json"), "r"))
    unet_config = json.load(open(os.path.join(model_dir, "unet/config.json"), "r"))
    vae_config = json.load(open(os.path.join(model_dir, "vae/config.json"), "r"))
    common_kwargs = {
        "path": model_dir,
        "embedding_dim": text_encoder_config["hidden_size"],
        "text_maxlen": text_encoder_config["max_position_embeddings"],
        "vae_scale_factor": 2 ** (len(vae_config["block_out_channels"]) - 1),
        "min_batch": min_batch_size, "max_batch": max_batch_size,
        "device": "cuda",
    }
    models = {
        "text_encoder": TextEncoder(name="text_encoder", **common_kwargs),
        "unet": UNet(name="unet", fp16=True, unet_dim=unet_config["in_channels"], **common_kwargs),
        # "vae": VAE(name="vae", **common_kwargs),
    }
    
    # Converting
    for i, (model_name, obj) in enumerate(models.items()):
        # Torch -> ONNX
        onnx_path = f"onnx/{args.data_name}/{model_name}.onnx"
        model = obj.get_model(args.data_name)
        min_shapes = obj.get_input_shape(min_batch_size, min_image_height, min_image_width)
        opt_shapes = obj.get_input_shape(opt_batch_size, opt_image_height, opt_image_width)
        max_shapes = obj.get_input_shape(max_batch_size, max_image_height, max_image_width)
        with torch.inference_mode(), torch.autocast("cuda"):
            inputs = obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
            torch.onnx.export(model,
                    inputs,
                    onnx_path,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=obj.get_input_names(),
                    output_names=obj.get_output_names(),
                    dynamic_axes=obj.get_dynamic_axes(),
            )
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # ONNX -> TensorRT
        command = f"""
            trtexec \
                --onnx={onnx_path} \
                --saveEngine=engine/{args.data_name}/{model_name}.plan \
                --minShapes={min_shapes} \
                --optShapes={opt_shapes} \
                --maxShapes={max_shapes} \
                --refit \
                --buildOnly \
                --fp16
            """
        exit_code = os.system(command)
        if exit_code != 0:
            raise Exception("Exit code", exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True)
    args = parser.parse_args()
    main(args)
