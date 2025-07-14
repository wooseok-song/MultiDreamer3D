import math
import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import imageio
import torch
import torch.nn as nn
import torchvision.transforms as T
import wandb
from arguments import ConceptParams, GenerateCamParams, GuidanceParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from torchvision.utils import save_image
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, tv_loss


def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    # TODO: add prenerg functions
    text_z_list = []
    weights_list = []
    K = 0
    # for b in range(azimuth):
    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt)
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0)  # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0)  # [B * K]
    return text_embeddings, weights


def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings["front"]
        end_z = embeddings["side"]
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings["front"], embeddings["side"]], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1 - r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings["side"]
        end_z = embeddings["back"]
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings["side"], embeddings["front"]], dim=0)
        front_neg_w = opt.negative_w
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)


def prepare_embeddings(guidance_opt, guidance, concept_opt, concept_dict):

    # Text embedding for base prompt
    base_prompt = concept_dict["base"]["step2_base_prompt"]

    embeddings = {}
    embeddings["default"] = guidance.get_text_embeds([base_prompt])
    embeddings["uncond"] = guidance.get_text_embeds([guidance_opt.negative])

    for d in ["front", "side", "back"]:
        embeddings[d] = guidance.get_text_embeds([f"{base_prompt}, {d} view"])
    embeddings["inverse_text"] = guidance.get_text_embeds(guidance_opt.inverse_text)

    concept_dict["base"]["text_embedding"] = embeddings


def prepare_embeddings_cism(guidance_opt, guidance, concept_opt, concept_dict):

    # Text embedding for base prompt
    base_prompt = concept_dict["base"]["step2_base_prompt"]
    base_embeddings = {}
    base_embeddings["default"] = guidance.get_text_embeds([base_prompt])
    base_embeddings["uncond"] = guidance.get_text_embeds([""])

    for d in ["front", "side", "back"]:
        base_embeddings[d] = guidance.get_text_embeds([f"{base_prompt}, {d} view"])
    base_embeddings["inverse_text"] = guidance.get_text_embeds(guidance_opt.inverse_text)
    concept_dict["base"]["base_text_embedding"] = base_embeddings

    # Text embedding for bg prompt
    bg_prompt = concept_dict["base"]["step2_bg_prompt"]
    bg_embeddings = {}
    bg_embeddings["default"] = guidance.get_text_embeds([bg_prompt])
    bg_embeddings["uncond"] = guidance.get_text_embeds([""])

    for d in ["front", "side", "back"]:
        bg_embeddings[d] = guidance.get_text_embeds([f"{bg_prompt}, {d} view"])
    bg_embeddings["inverse_text"] = guidance.get_text_embeds(guidance_opt.inverse_text)

    concept_dict["base"]["bg_text_embedding"] = bg_embeddings

    # Text embedding for concept prompt
    for i in range(concept_opt.concept_num):
        concept_key = f"concept{i}"
        embeddings = {}

        concept_prompt = concept_dict[concept_key]["step2_concept_prompt"]

        # text embeddings (stable-diffusion) and (IF)
        embeddings["default"] = guidance.get_text_embeds([concept_prompt])
        embeddings["uncond"] = guidance.get_text_embeds([guidance_opt.negative])

        for d in ["front", "side", "back"]:
            embeddings[d] = guidance.get_text_embeds([f"{concept_prompt}, {d} view"])
        embeddings["inverse_text"] = guidance.get_text_embeds(guidance_opt.inverse_text)

        concept_dict[concept_key]["text_embedding"] = embeddings


def prepare_embeddings_fedavg(guidance_opt, guidance, concept_opt, concept_dict):

    # Text embedding for base prompt
    base_prompt = concept_dict["base"]["step2_base_prompt"]
    base_embeddings = {}
    base_embeddings["default"] = guidance.get_text_embeds([base_prompt])
    base_embeddings["uncond"] = guidance.get_text_embeds([""])

    for d in ["front", "side", "back"]:
        base_embeddings[d] = guidance.get_text_embeds([f"{base_prompt}, {d} view"])
    base_embeddings["inverse_text"] = guidance.get_text_embeds(guidance_opt.inverse_text)
    concept_dict["fedavg"]["text_embedding"] = base_embeddings


def text_embedding_cism(guidance_opt, concept_opt, concept_dict, azimuth, text_concept_dict):

    base_text_embeddings = concept_dict["base"]["base_text_embedding"]
    text_z = [base_text_embeddings["uncond"]]

    if azimuth >= -90 and azimuth < 90:
        if azimuth >= 0:
            r = 1 - azimuth / 90
        else:
            r = 1 + azimuth / 90
        start_z = base_text_embeddings["front"]
        end_z = base_text_embeddings["side"]
    else:
        if azimuth >= 0:
            r = 1 - (azimuth - 90) / 90
        else:
            r = 1 + (azimuth + 90) / 90
        start_z = base_text_embeddings["side"]
        end_z = base_text_embeddings["back"]
    text_z.append(r * start_z + (1 - r) * end_z)
    text_z = torch.cat(text_z, dim=0)
    text_concept_dict["base"].append(text_z)

    # BG prompt text embedding
    bg_text_embeddings = concept_dict["base"]["bg_text_embedding"]
    text_z = [bg_text_embeddings["uncond"]]

    if azimuth >= -90 and azimuth < 90:
        if azimuth >= 0:
            r = 1 - azimuth / 90
        else:
            r = 1 + azimuth / 90
        start_z = bg_text_embeddings["front"]
        end_z = bg_text_embeddings["side"]
    else:
        if azimuth >= 0:
            r = 1 - (azimuth - 90) / 90
        else:
            r = 1 + (azimuth + 90) / 90
        start_z = bg_text_embeddings["side"]
        end_z = bg_text_embeddings["back"]
    text_z.append(r * start_z + (1 - r) * end_z)
    text_z = torch.cat(text_z, dim=0)
    text_concept_dict["bg"].append(text_z)

    # NOTE add concept promtps
    for j in range(concept_opt.concept_num):
        concept_key = f"concept{j}"

        concept_text_embeddings = concept_dict[concept_key]["text_embedding"]
        text_z = [concept_text_embeddings["uncond"]]

        if azimuth >= -90 and azimuth < 90:
            if azimuth >= 0:
                r = 1 - azimuth / 90
            else:
                r = 1 + azimuth / 90
            start_z = concept_text_embeddings["front"]
            end_z = concept_text_embeddings["side"]
        else:
            if azimuth >= 0:
                r = 1 - (azimuth - 90) / 90
            else:
                r = 1 + (azimuth + 90) / 90
            start_z = concept_text_embeddings["side"]
            end_z = concept_text_embeddings["back"]
        text_z.append(r * start_z + (1 - r) * end_z)

        text_z = torch.cat(text_z, dim=0)
        text_concept_dict[concept_key].append(text_z)


def text_embedding_fedavg(guidance_opt, concept_opt, concept_dict, azimuth, text_concept_dict):
    base_text_embeddings = concept_dict["fedavg"]["text_embedding"]
    text_z = [base_text_embeddings["uncond"]]

    if azimuth >= -90 and azimuth < 90:
        if azimuth >= 0:
            r = 1 - azimuth / 90
        else:
            r = 1 + azimuth / 90
        start_z = base_text_embeddings["front"]
        end_z = base_text_embeddings["side"]
    else:
        if azimuth >= 0:
            r = 1 - (azimuth - 90) / 90
        else:
            r = 1 + (azimuth + 90) / 90
        start_z = base_text_embeddings["side"]
        end_z = base_text_embeddings["back"]
    text_z.append(r * start_z + (1 - r) * end_z)
    text_z = torch.cat(text_z, dim=0)
    text_concept_dict["base"].append(text_z)
