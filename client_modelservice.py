from typing import List
import numpy as np
from PIL import Image

import tritonclient.http as httpclient


def create_reco_prompt(
    caption: str = '',
    phrases=[],
    boxes=[],
    normalize_boxes=True,
    w=512,
    h=512,
    num_bins=1000,
    ):
    """
    method to create ReCo prompt

    caption: global caption
    phrases: list of regional captions
    boxes: list of regional coordinates (unnormalized xyxy)
    """

    SOS_token = '<|startoftext|>'
    EOS_token = '<|endoftext|>'
    
    box_captions_with_coords = []
    
    box_captions_with_coords += [caption]
    box_captions_with_coords += [EOS_token]

    for phrase, box in zip(phrases, boxes):
        if normalize_boxes:
            x1, y1, x2, y2 = box
            box = [x1/w, y1/h, x2/w, y2/h]

        # quantize into bins
        quant_x0 = int(round((box[0] * (num_bins - 1))))
        quant_y0 = int(round((box[1] * (num_bins - 1))))
        quant_x1 = int(round((box[2] * (num_bins - 1))))
        quant_y1 = int(round((box[3] * (num_bins - 1))))
        
        # ReCo format
        # Add SOS/EOS before/after regional captions
        box_captions_with_coords += [
            f"<bin{str(quant_x0).zfill(3)}>",
            f"<bin{str(quant_y0).zfill(3)}>",
            f"<bin{str(quant_x1).zfill(3)}>",
            f"<bin{str(quant_y1).zfill(3)}>",
            SOS_token,
            phrase,
            EOS_token
        ]

    text = " ".join(box_captions_with_coords)
    return text


def send_request(
        client,
        mode: str,
        prompt: str,
        negative_prompt: str = "",
        image: Image.Image = None,
        mask_image: Image.Image = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        strength:float = 0.6,
        num_images_per_prompt:int = 4,
        seed: int = -1,
    ) -> List[Image.Image]:

    # Make numpy
    prompt = np.array([prompt], dtype=object)
    mode = np.array([mode], dtype=object)
    negative_prompt = np.array([negative_prompt], dtype=object)
    if image is None:
        image = np.zeros([512, 512, 3], dtype=np.uint8)
    else:
        image = np.array(image)
    if mask_image is None:
        mask_image = np.zeros([512, 512], dtype=np.uint8)
    else:
        mask_image = np.array(mask_image)
    num_inference_steps = np.array([num_inference_steps], dtype=np.int32)
    guidance_scale = np.array([guidance_scale], dtype=np.float32)
    strength = np.array([strength], dtype=np.float32)
    num_images_per_prompt = np.array([num_images_per_prompt], dtype=np.int32)
    seed = np.array([seed], dtype=np.int64)

    # Triton Inputs
    mode_in = httpclient.InferInput(name="mode", shape=mode.shape, datatype="BYTES")
    prompt_in = httpclient.InferInput(name="prompt", shape=prompt.shape, datatype="BYTES")
    negative_prompt_in = httpclient.InferInput(name="negative_prompt", shape=negative_prompt.shape, datatype="BYTES")
    image_in = httpclient.InferInput(name="image", shape=image.shape, datatype="UINT8")
    mask_image_in = httpclient.InferInput(name="mask_image", shape=mask_image.shape, datatype="UINT8")
    num_inference_steps_in = httpclient.InferInput("num_inference_steps", num_inference_steps.shape, "INT32")
    guidance_scale_in = httpclient.InferInput("guidance_scale", guidance_scale.shape, "FP32")
    strength_in = httpclient.InferInput("strength", strength.shape, "FP32")
    num_images_per_prompt_in = httpclient.InferInput("num_images_per_prompt", num_images_per_prompt.shape, "INT32")
    seed_in = httpclient.InferInput("seed", seed.shape, "INT64")

    mode_in.set_data_from_numpy(mode)
    prompt_in.set_data_from_numpy(prompt)
    negative_prompt_in.set_data_from_numpy(negative_prompt)
    image_in.set_data_from_numpy(image)
    mask_image_in.set_data_from_numpy(mask_image)
    num_inference_steps_in.set_data_from_numpy(num_inference_steps)
    guidance_scale_in.set_data_from_numpy(guidance_scale)
    strength_in.set_data_from_numpy(strength)
    num_images_per_prompt_in.set_data_from_numpy(num_images_per_prompt)
    seed_in.set_data_from_numpy(seed)

    output_img = httpclient.InferRequestedOutput("generated_image")

    # Inference
    query_response = client.infer(
        model_name="pipeline", 
        inputs=[
            mode_in,
            prompt_in,
            negative_prompt_in,
            image_in,
            mask_image_in,
            num_inference_steps_in,
            guidance_scale_in,
            strength_in,
            num_images_per_prompt_in,
            seed_in,
            ], 
        outputs=[output_img],
    )
    responses = query_response.as_numpy("generated_image")
    responses = [Image.fromarray((x*255).astype(np.uint8)) for x in responses]

    return responses


if __name__ == "__main__":
    client = httpclient.InferenceServerClient(url="0.0.0.0:8000")

    responses = send_request(
        client=client,
        mode="i2i",
        image=Image.open("sample.png"),
        mask_image=Image.open("sample_mask.png"),
        strength=0.4,
        prompt="a close up of a person's teeth. <|endoftext|> <bin000> <bin497> <bin183> <bin639> <|startoftext|> normal <|endoftext|> <bin743> <bin497> <bin909> <bin648> <|startoftext|> normal <|endoftext|> <bin711> <bin617> <bin886> <bin759> <|startoftext|> normal <|endoftext|> <bin001> <bin628> <bin212> <bin758> <|startoftext|> normal <|endoftext|> <bin683> <bin721> <bin818> <bin880> <|startoftext|> normal <|endoftext|> <bin105> <bin732> <bin241> <bin877> <|startoftext|> normal <|endoftext|> <bin591> <bin781> <bin707> <bin914> <|startoftext|> normal <|endoftext|> <bin225> <bin781> <bin343> <bin917> <|startoftext|> normal <|endoftext|> <bin318> <bin797> <bin468> <bin978> <|startoftext|> normal <|endoftext|> <bin460> <bin808> <bin618> <bin975> <|startoftext|> normal <|endoftext|> <bin771> <bin153> <bin985> <bin351> <|startoftext|> cavity <|endoftext|> <bin000> <bin169> <bin083> <bin366> <|startoftext|> cavity <|endoftext|> <bin760> <bin317> <bin961> <bin534> <|startoftext|> cavity <|endoftext|> <bin000> <bin328> <bin136> <bin540> <|startoftext|> cavity <|endoftext|>",
    )

    for i, im in enumerate(responses):
        im.save(f"{i}.png")
