from typing import List

import numpy as np
import tritonclient.http as httpclient
from PIL import Image, ImageDraw


def send_request(
        client,
        mode: str,
        prompt: str,
        boxes: List[List[float]],
        phrases: List[str],
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
    mode = np.array([mode], dtype=object)
    prompt = np.array([prompt], dtype=object)
    boxes = np.array(boxes, dtype=np.float32)
    phrases = np.array(phrases, dtype=object)
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
    boxes_in = httpclient.InferInput(name="boxes", shape=boxes.shape, datatype="FP32")
    phrases_in = httpclient.InferInput(name="phrases", shape=phrases.shape, datatype="BYTES")
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
    boxes_in.set_data_from_numpy(boxes)
    phrases_in.set_data_from_numpy(phrases)
    negative_prompt_in.set_data_from_numpy(negative_prompt)
    image_in.set_data_from_numpy(image)
    mask_image_in.set_data_from_numpy(mask_image)
    num_inference_steps_in.set_data_from_numpy(num_inference_steps)
    guidance_scale_in.set_data_from_numpy(guidance_scale)
    strength_in.set_data_from_numpy(strength)
    num_images_per_prompt_in.set_data_from_numpy(num_images_per_prompt)
    seed_in.set_data_from_numpy(seed)

    output_image = httpclient.InferRequestedOutput("output_image")
    output_boxes = httpclient.InferRequestedOutput("output_boxes")
    output_phrases = httpclient.InferRequestedOutput("output_phrases")

    # Inference
    query_response = client.infer(
        model_name="pipeline", 
        inputs=[
            mode_in,
            prompt_in,
            boxes_in,
            phrases_in,
            negative_prompt_in,
            image_in,
            # mask_image_in,
            num_inference_steps_in,
            guidance_scale_in,
            strength_in,
            num_images_per_prompt_in,
            seed_in,
            ], 
        outputs=[
            output_image,
            output_boxes,
            output_phrases,
        ],
    )

    out_images = query_response.as_numpy("output_image")
    out_images = [Image.fromarray((x*255).astype(np.uint8)) for x in out_images]

    out_boxes = query_response.as_numpy("output_boxes")
    out_phrases = query_response.as_numpy("output_phrases")

    return out_images, out_boxes, out_phrases


if __name__ == "__main__":
    client = httpclient.InferenceServerClient(url="0.0.0.0:8000")

    image_lst, boxes_lst, phrases_lst = send_request(
        client=client,
        mode="i2i",
        boxes=[
            [747, 60, 52.63, 361],
            [0, 148, 170.97, 248.28],
            [524, 106, 262.07, 365.19],
            [336, 140, 255.46, 326.58],
            [155, 210, 222.5, 232.09],
        ], # xywh
        phrases=[
            'cavity', 'cavity', 'cavity', 'cavity', 'cavity'
        ],
        image=Image.open("sample.png"),
        mask_image=Image.open("sample_mask.png"),
        strength=0.6,
        prompt="",
    )

    for i, (image, boxes, phrases) in enumerate(zip(image_lst, boxes_lst, phrases_lst)):
        # image_with_box = to_pil_image(draw_bounding_boxes(pil_to_tensor(image), torch.Tensor(boxes), labels=phrases))
        image_draw = ImageDraw.Draw(image)
        for box, phrase in zip(boxes, phrases):
            image_draw.rectangle(box, outline="red")
            image_draw.text((box[0], box[1]), phrase.decode())
        image.save(f"{i}.png")
