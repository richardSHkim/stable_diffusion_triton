import torch

from ..utils import create_reco_prompt


def process_t2i(
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
    generator=torch.Generator(device=pipe.device).manual_seed(seed) if seed >=0 else None

    # Get ReCo prompt
    prompt = create_reco_prompt(caption=prompt, phrases=phrases, boxes=boxes, normalize_boxes=True, w=width, h=height)

    results = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="np",
        ).images
    return results