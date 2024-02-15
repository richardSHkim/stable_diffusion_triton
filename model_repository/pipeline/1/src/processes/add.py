import numpy as np
import torch
from PIL import Image

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image

from ..utils import create_reco_prompt


def process_add(
        pipe,
        data_name,
        mode,
        class_name_to_inpaint,
        raw_image,
        mask_image,
        boxes,
        phrases,
        prompt,
        negative_prompt,
        num_inference_steps,
        guidance_scale,
        strength,
        strength_2,
        num_images_per_prompt,
        seed,
        width,
        height,
    ):

    # only use first layer mask
    mask = np.array(mask_image)
    y1 = np.min(np.where(mask == 255)[0])
    y2 = np.max(np.where(mask == 255)[0])
    x1 = np.min(np.where(mask == 255)[1])
    x2 = np.max(np.where(mask == 255)[1])
    boxes = [[x1, y1, x2, y2]]
    phrases = [class_name_to_inpaint]
    
    is_margin = False
    is_crop = False
    is_new = False

    device = pipe.device

    # pick a box
    filtered_boxes = []
    filtered_phrases = []
    for phrase, box in zip(phrases, boxes):
        if phrase == class_name_to_inpaint:
            filtered_boxes.append(box)
            filtered_phrases.append(phrase)
    boxes = filtered_boxes
    phrases = filtered_phrases

    # mask
    mask = np.zeros_like(np.array(raw_image))
    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box]

        if is_margin:
            margin_x = (x2 - x1) // 4
            margin_y = (y2 - y1) // 4
            y1 = max(y1 - margin_y, 0)
            y2 = min(y2 + margin_y, raw_image.height)
            x1 = max(x1 - margin_x, 0)
            x2 = min(x2 + margin_x, raw_image.width)

        mask[y1:y2, x1:x2, :] = 255

    # crop
    if is_crop:    
        margin_x = (x2 - x1) // 2
        margin_y = (y2 - y1) // 2
        crop_y1 = max(y1 - margin_y, 0)
        crop_y2 = min(y2 + margin_y, raw_image.height)
        crop_x1 = max(x1 - margin_x, 0)
        crop_x2 = min(x2 + margin_x, raw_image.width)
        cropped_image = Image.fromarray(np.array(raw_image)[crop_y1:crop_y2, crop_x1:crop_x2, :])
        cropped_mask = Image.fromarray(mask[crop_y1:crop_y2, crop_x1:crop_x2, :])

        # simple resize
        image_to_inpaint = cropped_image.resize((width, height))
        mask_to_inpaint = cropped_mask.resize((width, height))

        # resize and fill
        # image_to_inpaint = resize_image(2, cropped_image, width, height)

        mask_to_inpaint = Image.fromarray((np.array(mask_to_inpaint) > 255*0.5).astype(np.uint8)*255)
    else:
        image_to_inpaint = raw_image
        mask_to_inpaint = Image.fromarray(mask)

    if is_new:
        generator=torch.Generator(device=device).manual_seed(seed) if seed >=0 else None
        
        shape = (num_images_per_prompt, pipe.vae.config.latent_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)

        _, _, image_ = prepare_mask_and_masked_image(image_to_inpaint, mask_to_inpaint, height, width, return_image=True)

        image_ = image_.to(device, dtype=pipe.unet.dtype)
        image_latent = pipe._encode_vae_image(image=image_, generator=generator)

        # mask latent
        latmask = mask_to_inpaint.convert('RGB').resize((image_latent.shape[3], image_latent.shape[2]))
        latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
        latmask = latmask[0]
        latmask = np.around(latmask)
        latmask = np.tile(latmask[None], (4, 1, 1))

        mask_ = torch.asarray(1.0 - latmask).to(device).type(pipe.unet.dtype)
        nmask_ = torch.asarray(latmask).to(device).type(pipe.unet.dtype)

        image_latent = image_latent * mask_ + randn_tensor(shape, generator=generator, device=pipe._execution_device, dtype=pipe.unet.dtype) * nmask_

        # add noise
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, _ = pipe.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        latent_timestep = timesteps[:1].repeat(num_images_per_prompt)
        noise = randn_tensor(shape, generator=generator, device=pipe._execution_device, dtype=pipe.unet.dtype)
        init_latent = pipe.scheduler.add_noise(image_latent, noise, latent_timestep)
    else:
        init_latent = None

    if is_crop:
        mask_ = np.where(np.array(mask_to_inpaint) == 255)
        box = [np.min(mask_[1]), np.min(mask_[0]), np.max(mask_[1]), np.max(mask_[0])]

    # random seed
    generator=torch.Generator(device=device).manual_seed(seed) if seed >=0 else None

    # Get ReCo prompt
    prompt = create_reco_prompt(caption=prompt, phrases=phrases, boxes=boxes, normalize_boxes=True, w=width, h=height)

    # run
    results = pipe.call_inpaint(prompt=prompt, 
                    negative_prompt=negative_prompt,
                    image=image_to_inpaint, 
                    mask_image=mask_to_inpaint,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps, 
                    strength=strength,
                    strength_2=strength_2,
                    latents=init_latent,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    output_type="np",
                    ).images
    return results