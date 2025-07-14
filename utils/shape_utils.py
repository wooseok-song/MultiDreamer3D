import base64
import io
import os
from io import BytesIO

import numpy as np
import open3d as o3d
import openai
import requests
import torch
from PIL import Image
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint
from point_e.util.plotting import plot_point_cloud
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_config, load_model
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh, gif_widget
from tqdm.auto import tqdm


def init_from_shape(prompt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("creating base model...")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    model.load_state_dict(
        torch.load("./load/shapE_finetuned_with_825kdata.pth", map_location=device)["model_state_dict"]
    )
    diffusion = diffusion_from_config_shape(load_config("diffusion"))

    batch_size = 1
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    render_mode = "nerf"  # you can change this to 'stf'
    size = 256  # this is the size of the renders; higher values take longer to render.

    cameras = create_pan_cameras(size, device)

    pc = decode_latent_mesh(xm, latents[0]).tri_mesh()

    skip = 1
    coords = pc.verts
    rgb = np.concatenate(
        [pc.vertex_channels["R"][:, None], pc.vertex_channels["G"][:, None], pc.vertex_channels["B"][:, None]],
        axis=1,
    )

    xyz = coords[::skip]
    rgb = rgb[::skip]

    sample_indices = np.random.choice(rgb.shape[0], 4096, replace=False)

    xyz = xyz[sample_indices]
    rgb = rgb[sample_indices]

    return xyz, rgb


def init_from_shape_candidates(concept_opts, prompt, path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("creating base model...")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    model.load_state_dict(
        torch.load("./load/shapE_finetuned_with_825kdata.pth", map_location=device)["model_state_dict"]
    )
    diffusion = diffusion_from_config_shape(load_config("diffusion"))

    batch_size = 8
    guidance_scale = 15.0
    print("prompt", prompt)
    # torch.random.manual_seed(0)
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    render_mode = "nerf"  # you can change this to 'stf'
    size = 128  # this is the size of the renders; higher values take longer to render.

    cameras = create_pan_cameras(size, device)

    images_list = []
    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        images[0].save(os.path.join(path, f"cand_{i}.png"))
        images_list.append(images[0])

    idx_select = _select_candidate_llm(concept_opts, prompt, images_list)

    images_list[idx_select].save(os.path.join(path, f"selected_cand.png"))

    pc = decode_latent_mesh(xm, latents[idx_select]).tri_mesh()

    skip = 1
    coords = pc.verts
    rgb = np.concatenate(
        [pc.vertex_channels["R"][:, None], pc.vertex_channels["G"][:, None], pc.vertex_channels["B"][:, None]],
        axis=1,
    )

    xyz = coords[::skip]
    rgb = rgb[::skip]

    sample_indices = np.random.choice(rgb.shape[0], 4096, replace=False)

    xyz = xyz[sample_indices]
    rgb = rgb[sample_indices]

    return xyz, rgb


def _select_candidate_llm(concept_opts, prompt, images):

    def encode_image(pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_byte = buffered.getvalue()
        return base64.b64encode(img_byte).decode("utf-8")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {concept_opts.gpt_key}"}
    templatev0_1 = f"""I will give an text prompt and images.
    Your task is to select which image most well depicts the text prompt.
    Just give an index of images.
    The index range is [0,7]
    
    Text prompt : {prompt}, front view.
    """

    base64_image_list = [encode_image(images[i]) for i in range(8)]

    payload = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": templatev0_1,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_list[0]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_list[1]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_list[2]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_list[3]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_list[4]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_list[5]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_list[6]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_list[7]}"}},
                ],
            }
        ],
        "max_tokens": 1000,
        "seed": 0,
        "temperature": 0.0,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    content = response.json()["choices"][0]["message"]["content"]

    selected_content = int(content)
    print(f"LLM selected samples :{selected_content}")

    return selected_content
