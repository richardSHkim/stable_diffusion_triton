from typing import List
import numpy as np
from PIL import Image
from multiprocessing import Process, Manager, Lock

import tritonclient.http as httpclient


def send_request_mp(
        url: str,
        num_results: int,
        **kwargs,
    ):
    client = httpclient.InferenceServerClient(url=url)
    manager = Manager()
    results = manager.dict()
    results["results"] = []
    lock = Lock()

    processes = []
    for _ in range(num_results):
        p = Process(target=send_request, args=(results, lock, client), kwargs=kwargs)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return results["results"]


def send_request(
        results,
        lock,
        client,
        data_name: str,
        mode: str,
        class_name_to_inpaint: str,
        raw_image: Image.Image,
        mask_image: Image.Image,
        boxes: List[List[float]],
        phrases: List[str],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.7,
        strength_2: float = 0.0,
        num_images_per_prompt: int = 1,
        seed: int = -1,
    ) -> List[Image.Image]:

    # Make numpy
    data_name = np.array([data_name], dtype=object)
    mode = np.array([mode], dtype=object)
    class_name_to_inpaint = np.array([class_name_to_inpaint], dtype=object)
    raw_image = np.array(raw_image, dtype=np.uint8)
    mask_image = np.array(mask_image, dtype=np.uint8)
    boxes = np.array(boxes, dtype=np.float32)
    phrases = np.array(phrases, dtype=object)
    prompt = np.array([prompt], dtype=object)
    negative_prompt = np.array([negative_prompt], dtype=object)
    num_inference_steps = np.array([num_inference_steps], dtype=np.int32)
    guidance_scale = np.array([guidance_scale], dtype=np.float32)
    strength = np.array([strength], dtype=np.float32)
    strength_2 = np.array([strength_2], dtype=np.float32)
    num_images_per_prompt = np.array([num_images_per_prompt], dtype=np.int32)
    seed = np.array([seed], dtype=np.int64)

    # Triton Inputs
    data_name_in = httpclient.InferInput(name="data_name", shape=data_name.shape, datatype="BYTES")
    mode_in = httpclient.InferInput(name="mode", shape=mode.shape, datatype="BYTES")
    class_name_to_inpaint_in = httpclient.InferInput(name="class_name_to_inpaint", shape=class_name_to_inpaint.shape, datatype="BYTES")
    raw_image_in = httpclient.InferInput(name="raw_image", shape=raw_image.shape, datatype="UINT8")
    mask_image_in = httpclient.InferInput(name="mask_image", shape=mask_image.shape, datatype="UINT8")
    boxes_in = httpclient.InferInput(name="boxes", shape=boxes.shape, datatype="FP32")
    phrases_in = httpclient.InferInput(name="phrases", shape=phrases.shape, datatype="BYTES")
    prompt_in = httpclient.InferInput(name="prompt", shape=prompt.shape, datatype="BYTES")
    negative_prompt_in = httpclient.InferInput(name="negative_prompt", shape=negative_prompt.shape, datatype="BYTES")
    num_inference_steps_in = httpclient.InferInput("num_inference_steps", num_inference_steps.shape, "INT32")
    guidance_scale_in = httpclient.InferInput("guidance_scale", guidance_scale.shape, "FP32")
    strength_in = httpclient.InferInput("strength", strength.shape, "FP32")
    strength_2_in = httpclient.InferInput("strength_2", strength_2.shape, "FP32")
    num_images_per_prompt_in = httpclient.InferInput("num_images_per_prompt", num_images_per_prompt.shape, "INT32")
    seed_in = httpclient.InferInput("seed", seed.shape, "INT64")

    data_name_in.set_data_from_numpy(data_name)
    mode_in.set_data_from_numpy(mode)
    class_name_to_inpaint_in.set_data_from_numpy(class_name_to_inpaint)
    raw_image_in.set_data_from_numpy(raw_image)
    mask_image_in.set_data_from_numpy(mask_image)
    boxes_in.set_data_from_numpy(boxes)
    phrases_in.set_data_from_numpy(phrases)
    prompt_in.set_data_from_numpy(prompt)
    negative_prompt_in.set_data_from_numpy(negative_prompt)
    num_inference_steps_in.set_data_from_numpy(num_inference_steps)
    guidance_scale_in.set_data_from_numpy(guidance_scale)
    strength_in.set_data_from_numpy(strength)
    strength_2_in.set_data_from_numpy(strength_2)
    num_images_per_prompt_in.set_data_from_numpy(num_images_per_prompt)
    seed_in.set_data_from_numpy(seed)

    output_img = httpclient.InferRequestedOutput("generated_image")

    # Inference
    query_response = client.infer(
        model_name="pipeline", 
        inputs=[data_name_in,
                mode_in,
                class_name_to_inpaint_in,
                raw_image_in,
                mask_image_in,
                boxes_in,
                phrases_in,
                prompt_in,
                negative_prompt_in,
                num_inference_steps_in,
                guidance_scale_in,
                strength_in,
                strength_2_in,
                num_images_per_prompt_in,
                seed_in,
                ], 
        outputs=[output_img],
    )
    responses = query_response.as_numpy("generated_image")
    responses = [Image.fromarray((x*255).astype(np.uint8)) for x in responses]

    lock.acquire()
    results["results"] += responses
    lock.release()


if __name__ == "__main__":
    kwargs = {
        "url": "0.0.0.0:8000",
        "num_results": 4,
        "data_name": "roboflow-cable-damage",
        "mode": "Text-to-Image",
        "class_name_to_inpaint": "break",
        "raw_image": Image.fromarray(np.zeros([512, 512, 3]).astype(np.uint8)),
        "mask_image": Image.fromarray(np.zeros([512, 512]).astype(np.uint8)),
        "boxes": [[0, 0, 100, 100]],
        "phrases": ["break"],
        "prompt": "",
        "negative_prompt": "",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "strength": 0.7,
        "strength_2": 0.0,
        "seed": -1,
    }

    results = send_request_mp(**kwargs)

    for i, im in enumerate(results):
        im.save(f"{i}.png")
