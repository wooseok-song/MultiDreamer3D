import ast
import base64
import copy
import io
import json
import os
import pickle
import random
import re
from io import BytesIO
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import openai
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from gaussian_renderer import network_gui, render
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from rembg import remove
from scene import GaussianModel, Scene
from scene.dataset_readers import (
    GenerateCircleCameras,
    GeneratePurnCameras,
    GenerateRandomCameras,
    GenerateRefineCameras,
    sceneLoadTypeCallbacks,
)
from tqdm import tqdm
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos, cameraList_from_RcamInfos


def extract_objects_coordinates(output):

    pattern = re.compile(r"\d+\.\s*\((\w+), \[(.*?)\]\)")
    matches = pattern.findall(output)

    objects_coordinates = {}

    for match in matches:
        obj_name = match[0]
        coordinates = list(map(int, match[1].split(", ")))
        objects_coordinates[obj_name] = coordinates

    return objects_coordinates


def step1_3D_layout_controller(concept_dict, concept_opt, save_folder):

    prompt = concept_dict["base"]["step2_base_prompt"]

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {concept_opt.gpt_key}"}

    templatev0_1 = f"""You are an intelligent bounding box generator. 
    I will provide you with a caption for a 3D scene. 
    Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene at last. 
    The 3D scenes are of size 512x512x512. 
    The top right back corner has coordinates [512, 512 ,512]. 
    The bottom left front corner has coordinate [0, 0, 0]. 
    center of bounding box is [256,256,256]. 
    And the coordinate system is [x,y,z] The bounding boxes should not go beyond the image boundaries. 
    Each bounding box should be in the format of (object name, [bottom x coordinate, bottom y coordinate, bottom z coordinate,  box width, box depth, box height]) and should not include more than one object. 
    Do not put objects that are already provided in the bounding boxes into the background prompt. 
    Do not include non-existing or excluded objects in the background prompt. 
    Use "A realistic scene" as the background prompt if no background is given in the prompt. 
    If needed, you can make reasonable guesses. Please never reproduce the bounding box of example.
    Please generate diverse bounding box as possible.

    ---Example---

    Caption: A dog wearing wearable_sunglasses
    Objects: 1.(dog, [156, 80, 150, 200, 340, 200]) 2.(wearable_sunglasses, [200, 101, 280, 101, 130, 50])

    Caption: A bear_plushie wearing pink_sunglasses
    Objects: 1.(bear_plushie, [128, 81, 123, 263, 302, 241]) 2.(pink_sunglasses, [163, 105, 282, 191, 147, 63])

    Caption: A cat wearing headphone
    Objects: 1.(cat, [156, 82, 137, 202, 304, 207]) 2.(headphone, [171, 99, 242, 171, 123, 117 ])

    Caption: A plushie_panguin wearing headphone
    Objects: 1.(plushie_panguin, [156, 140, 140, 202, 253, 204]) 2.(headphone, [166, 160, 250, 180, 190, 100])

    Caption: A dog_1 on left and dog_2 on right
    Objects: 1.(dog_1, [100, 120, 130, 140, 300, 202] ) 2.(dog_2, [262, 123, 136, 154, 302, 221])

    Caption: A robot_toy on left and plushie_penguin on right
    Objects: 1.(robot,[80, 154, 140, 160, 192, 160] ) 2.(plushie_penguin, [280, 140, 140, 167, 204, 180])

    Caption: A dog on the table
    Objects: 1.(dog, [166, 150, 220, 177, 223, 142]) 2.(table, [126, 101, 120, 280, 330, 123])

    Caption: A poop_emoji on the table
    Objects: 1.(poop_emoji, [166, 149, 240, 180, 221, 140]) 2.(table, [126, 100, 153, 282, 330, 113])

    Caption: A cat sitting on car.
    Objects: 1.(cat, [187, 100, 176, 143, 200, 142]) 2.(car, [101, 32, 130, 297, 322, 90])

    Caption: A dog jumping on the left and cat jumping on the right.
    Objects: 1.(dog, [100, 200, 230, 160, 300, 220]) 2.(cat, [300, 200, 230, 141, 303, 201])

    Caption: A dog with a blue tie on the left and dog with a red tie on the right.
    Objects: 1.(dog, [100, 118, 130, 140, 302, 200]) 2.(dog, [262, 121, 136, 154, 302, 218])

    Caption: A dog in a suit on the left and dog in a suit on the right.
    Objects: 1.(dog, [80, 110, 130, 150, 270, 220]) 2.(dog2, [272, 110, 130, 160, 270, 230])

    Caption: A dog on the left and plushie_penguin on the middle and robot_toy on the right.
    Objects: 1.(dog, [80, 120, 150, 150, 250, 200]) 2.(plushie_penguin, [226, 140, 150, 160, 230, 190]) 3.(robot_toy, [350, 154, 150, 140, 200, 160])

    Caption: A bear_plushie doll on the left and dog1 on the middle and dog2 on the right.
    Objects: 1.(bear_plushie, [56, 112, 112, 144, 288, 192]) 2.(dog1, [200, 132, 132, 160, 240, 196]) 3.(dog2, [368, 150, 151, 136, 256, 184])

    Caption: A cat on the left and dog at the back and dog2 on the right.
    Objects: 1.(cat, [77, 120, 150, 150, 250, 201]) 2.(dog, [226, 250, 150, 160, 233, 194]) 3.(dog2, [350, 120, 147, 140, 202, 160])

    Caption: {prompt}
    """

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
                ],
            }
        ],
        "max_tokens": 1000,
        "seed": 0,
        "temperature": 0.0,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    content = response.json()["choices"][0]["message"]["content"]
    print("output : ", content)
    content_dict = extract_objects_coordinates(content)

    # NOTE CHATGPT OUTPUTS different outputs that might not aligned with concept prompts.
    for i, (key) in enumerate(content_dict):
        concept_key = f"concept{i}"
        concept_dict[concept_key]["step1_bbox"] = content_dict[key]

    visualize_3d_layout(content_dict, prompt, save_folder)


def set_3d_layout(concept_dict, concept_opt, bound=2.0, normalization_value=512):
    if concept_opt.step1_layout:
        print("Setting 3D layout: 3D BBOX")
        for i in range(concept_opt.concept_num):
            concept_key = f"concept{i}"

            x, y, z, width, depth, height = concept_dict[concept_key]["step1_bbox"]
            s_max = bound * max(width, height) / normalization_value
            s_min = bound * min(width, height) / normalization_value
            tx = bound * ((x + width / 2 - normalization_value / 2) / (normalization_value / 2))
            ty = bound * ((y + depth / 2 - normalization_value / 2) / (normalization_value / 2))
            tz = bound * ((z + height / 2 - normalization_value / 2) / (normalization_value / 2))

            concept_dict[concept_key]["step1_transform"] = (
                torch.tensor([s_min, tx, ty, tz, 0], dtype=torch.float32).cuda().requires_grad_(False)
            )

    else:
        print("Setting 3D layout: Center")
        for i in range(concept_opt.concept_num):
            concept_key = f"concept{i}"
            concept_dict[concept_key]["step1_transform"] = torch.tensor([1, 0, 0, 0, 0]).cuda().requires_grad_(False)


def visualize_3d_layout(content_dict, prompt, save_folder):

    fig = plt.figure(figsize=(20, 8))
    ax_3d = fig.add_subplot(131, projection="3d")
    ax_front = fig.add_subplot(132)
    ax_side = fig.add_subplot(133)

    scene_size = (512, 512, 512)

    objects = content_dict
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(objects))]
    co = copy.deepcopy(colors)

    for obj, details in objects.items():

        x, y, z = details[:3]
        dx, dy, dz = details[3:]
        color = co.pop(0)
        ax_3d.bar3d(x, y, z, dx, dy, dz, color=color, alpha=0.3, edgecolor=color, label=obj)

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")

    ax_3d.set_xlim(scene_size[0], 0)
    ax_3d.set_ylim(scene_size[1], 0)
    ax_3d.set_zlim(0, scene_size[2])

    ax_3d.view_init(elev=30, azim=120)
    co = copy.deepcopy(colors)

    for obj, details in objects.items():
        x, y, z = details[:3]
        dx, dy, dz = details[3:]
        color = co.pop(0)
        ax_front.add_patch(plt.Rectangle((x, z), dx, dz, color=color, alpha=0.3, label=obj))

    ax_front.set_xlabel("X")
    ax_front.set_ylabel("Z")

    ax_front.set_xlim(0, scene_size[0])
    ax_front.set_ylim(0, scene_size[2])

    co = copy.deepcopy(colors)

    for obj, details in objects.items():
        x, y, z = details[:3]
        dx, dy, dz = details[3:]
        color = co.pop(0)
        ax_side.add_patch(plt.Rectangle((y, z), dy, dz, color=color, alpha=0.3, label=obj))

    ax_side.set_xlabel("Y")
    ax_side.set_ylabel("Z")

    ax_side.set_xlim(0, scene_size[1])
    ax_side.set_ylim(0, scene_size[2])

    ax_3d.legend(loc="upper right")
    ax_front.legend(loc="upper right")
    ax_side.legend(loc="upper right")

    ax_3d.set_title(f"3D Bounding Boxes Visualization\n prompt:{prompt}")
    ax_front.set_title("Front View Bounding Boxes")
    ax_side.set_title("Side View Bounding Boxes")

    plt.tight_layout()
    plt.savefig(f"{save_folder}/3D_llm.png")
    plt.close()
