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
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        seed: int = -1,
    ) -> List[Image.Image]:

    # Make numpy
    prompt = np.array([prompt], dtype=object)
    negative_prompt = np.array([negative_prompt], dtype=object)
    num_inference_steps = np.array([num_inference_steps], dtype=np.int32)
    guidance_scale = np.array([guidance_scale], dtype=np.float32)
    seed = np.array([seed], dtype=np.int64)

    # Triton Inputs
    prompt_in = httpclient.InferInput(name="prompt", shape=prompt.shape, datatype="BYTES")
    negative_prompt_in = httpclient.InferInput(name="negative_prompt", shape=negative_prompt.shape, datatype="BYTES")
    num_inference_steps_in = httpclient.InferInput("num_inference_steps", num_inference_steps.shape, "INT32")
    guidance_scale_in = httpclient.InferInput("guidance_scale", guidance_scale.shape, "FP32")
    seed_in = httpclient.InferInput("seed", seed.shape, "INT64")

    prompt_in.set_data_from_numpy(prompt)
    negative_prompt_in.set_data_from_numpy(negative_prompt)
    num_inference_steps_in.set_data_from_numpy(num_inference_steps)
    guidance_scale_in.set_data_from_numpy(guidance_scale)
    seed_in.set_data_from_numpy(seed)

    output_img = httpclient.InferRequestedOutput("generated_image")

    # Inference
    query_response = client.infer(
        model_name="pipeline", 
        inputs=[prompt_in,
                negative_prompt_in,
                num_inference_steps_in,
                guidance_scale_in,
                seed_in,
                ], 
        outputs=[output_img],
    )
    responses = query_response.as_numpy("generated_image")
    responses = [Image.fromarray((x*255).astype(np.uint8)) for x in responses]

    return responses


if __name__ == "__main__":
    client = httpclient.InferenceServerClient(url="0.0.0.0:8010")

    responses = send_request(
        client=client,
        prompt="a close up of a person's teeth. <|endoftext|> <bin000> <bin497> <bin183> <bin639> <|startoftext|> normal <|endoftext|> <bin743> <bin497> <bin909> <bin648> <|startoftext|> normal <|endoftext|> <bin711> <bin617> <bin886> <bin759> <|startoftext|> normal <|endoftext|> <bin001> <bin628> <bin212> <bin758> <|startoftext|> normal <|endoftext|> <bin683> <bin721> <bin818> <bin880> <|startoftext|> normal <|endoftext|> <bin105> <bin732> <bin241> <bin877> <|startoftext|> normal <|endoftext|> <bin591> <bin781> <bin707> <bin914> <|startoftext|> normal <|endoftext|> <bin225> <bin781> <bin343> <bin917> <|startoftext|> normal <|endoftext|> <bin318> <bin797> <bin468> <bin978> <|startoftext|> normal <|endoftext|> <bin460> <bin808> <bin618> <bin975> <|startoftext|> normal <|endoftext|> <bin771> <bin153> <bin985> <bin351> <|startoftext|> cavity <|endoftext|> <bin000> <bin169> <bin083> <bin366> <|startoftext|> cavity <|endoftext|> <bin760> <bin317> <bin961> <bin534> <|startoftext|> cavity <|endoftext|> <bin000> <bin328> <bin136> <bin540> <|startoftext|> cavity <|endoftext|>",
    )

    for i, im in enumerate(responses):
        im.save(f"{i}.png")
