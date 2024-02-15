import numpy as np
import torch
from PIL import Image

from ..utils import create_reco_prompt


def process_outpaint(
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
    
    is_margin = False

    device = pipe.device

    # mask
    mask = np.ones_like(np.array(raw_image)) * 255
    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box]

        if is_margin:
            margin_x = (x2 - x1) // 4
            margin_y = (y2 - y1) // 4
            y1 = max(y1 - margin_y, 0)
            y2 = min(y2 + margin_y, raw_image.height)
            x1 = max(x1 - margin_x, 0)
            x2 = min(x2 + margin_x, raw_image.width)

        mask[y1:y2, x1:x2, :] = 0

    image_to_inpaint = raw_image
    mask_to_inpaint = Image.fromarray(mask)

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
                    latents=None,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    output_type="np",
                    ).images
    return results