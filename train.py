#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import wandb
from arguments import ConceptParams, GenerateCamParams, GuidanceParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render, render_concept
from layout.layout_generator_utils import set_3d_layout, step1_3D_layout_controller
from parsing.concept_parsing_utils import concept_prompt_parsing
from prompt.prompt_utils import (
    prepare_embeddings,
    prepare_embeddings_cism,
    prepare_embeddings_fedavg,
    text_embedding_cism,
    text_embedding_fedavg,
)
from scene import GaussianModel, Scene
from torchvision.utils import save_image
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, tv_loss

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

"""
===============================================
PUT YOUR GPT API KEY HERE
===============================================
"""
GPT_API_KEY = "YOUR GPT API KEY"


def concept_dict_setup(concept_dict, concept_opt):

    concept_dict["base"] = {}
    concept_dict["base"]["step1_base_prompt"] = getattr(concept_opt, "step1_base_prompt")
    concept_dict["base"]["step2_base_prompt"] = getattr(concept_opt, "step2_base_prompt")
    concept_dict["base"]["step2_bg_prompt"] = getattr(concept_opt, "step2_bg_prompt")

    # Generate random
    for i in range(concept_opt.concept_num):
        concept_key = f"concept{i}"
        concept_dict[concept_key] = {}
        concept_dict[concept_key]["step1_prompt"] = getattr(concept_opt, f"step1_concept{i}_prompt")
        concept_dict[concept_key]["step1_bbox"] = getattr(concept_opt, f"step1_concept{i}_bbox")
        concept_dict[concept_key]["step1_transform"] = getattr(concept_opt, f"step1_concept{i}_transform")
        concept_dict[concept_key]["step1_pcd_path"] = ""
        concept_dict[concept_key]["step1_pcd"] = ""
        concept_dict[concept_key]["step1_pcd_rgb"] = ""
        concept_dict[concept_key]["step2_lora_path"] = getattr(concept_opt, f"step2_LoRA_concept{i}_path")
        concept_dict[concept_key]["step2_concept_prompt"] = getattr(concept_opt, f"step2_concept{i}_prompt")
        concept_dict[concept_key]["real_data_path"] = getattr(concept_opt, f"real_data_concept{i}_path")
    print("============== Concept dict setup complete ==============")


def guidance_setup(guidance_opt, concept_opt, concept_dict):

    if guidance_opt.guidance == "SD":
        if concept_opt.concept_type == "CISM":
            from guidance.sd_utils_cism import StableDiffusionCISM

            guidance = StableDiffusionCISM(
                guidance_opt.g_device,
                guidance_opt.fp16,
                guidance_opt.vram_O,
                guidance_opt.t_range,
                guidance_opt.max_t_range,
                num_train_timesteps=guidance_opt.num_train_timesteps,
                ddim_inv=guidance_opt.ddim_inv,
                textual_inversion_path=guidance_opt.textual_inversion_path,
                LoRA_path=guidance_opt.LoRA_path,
                guidance_opt=guidance_opt,
                concept_opt=concept_opt,
                concept_dict=concept_dict,
            )
            prepare_embeddings_cism(guidance_opt, guidance, concept_opt, concept_dict)
            print("Guidance: Stable diffsuion with CISM")
        elif concept_opt.concept_type == "FEDAVG":
            from guidance.sd_utils_cism import StableDiffusionCISM

            guidance = StableDiffusionCISM(
                guidance_opt.g_device,
                guidance_opt.fp16,
                guidance_opt.vram_O,
                guidance_opt.t_range,
                guidance_opt.max_t_range,
                num_train_timesteps=guidance_opt.num_train_timesteps,
                ddim_inv=guidance_opt.ddim_inv,
                textual_inversion_path=guidance_opt.textual_inversion_path,
                LoRA_path=guidance_opt.LoRA_path,
                guidance_opt=guidance_opt,
                concept_opt=concept_opt,
                concept_dict=concept_dict,
            )
            prepare_embeddings_fedavg(guidance_opt, guidance, concept_opt, concept_dict)
            print("Guidance: Stable diffsuion with FedAVG")
        else:
            from guidance.sd_utils import StableDiffusion

            guidance = StableDiffusion(
                guidance_opt.g_device,
                guidance_opt.fp16,
                guidance_opt.vram_O,
                guidance_opt.t_range,
                guidance_opt.max_t_range,
                num_train_timesteps=guidance_opt.num_train_timesteps,
                ddim_inv=guidance_opt.ddim_inv,
                textual_inversion_path=guidance_opt.textual_inversion_path,
                LoRA_path=guidance_opt.LoRA_path,
                guidance_opt=guidance_opt,
                concept_opt=concept_opt,
                concept_dict=concept_dict,
            )
            prepare_embeddings(guidance_opt, guidance, concept_opt, concept_dict)
            print("Stable diffsuion baseline loaded")

    else:
        raise ValueError(f"{guidance_opt.guidance} not supported.")
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False

    return guidance


def training(
    dataset,
    opt,
    pipe,
    gcams,
    guidance_opt,
    concept_opt,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    save_video,
    wandb_logger,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    # NOTE This is to ensure concept masks doesn't have  background color
    bg_color += [0] * 5
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)

    # Initialize concept dict and Normalize 3D BBOX
    concept_dict = {}
    concept_dict_setup(concept_dict, concept_opt)
    set_3d_layout(concept_dict, concept_opt, normalization_value=512)

    # Initialize 3D Gaussian with scene
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gcams, concept_opt, gaussians, concept_dict)

    if concept_opt.step1_3d_layout_generator:
        save_llm_path = os.path.join(dataset._model_path, "save_llm/")
        if not os.path.exists(save_llm_path):
            os.makedirs(save_llm_path)  # makedirs
        # 3D Layout Generator, Updating 3D bbox with LLM generated 3D bbox
        print("========================== Generating 3D Layouts with 3D Layout Generator ==========================")
        # Generate 3D bbox with 3D layout controller
        step1_3D_layout_controller(concept_dict, concept_opt, save_llm_path)

        # Update 3D layout with LLM generated 3D bbox
        set_3d_layout(concept_dict, concept_opt, normalization_value=512)
        scene.update_gaussian_with_3d_layouts(concept_dict, concept_opt)

    if concept_opt.step2_auto_concept_parsing:
        print("========================== Automatic concept prompt parsing ==========================")
        concept_prompt_parsing(concept_dict, concept_opt)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    save_folder = os.path.join(dataset._model_path, "train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs
        print("train_process is in :", save_folder)

    # set up pretrain diffusion models and text_embedings
    guidance = guidance_setup(guidance_opt, concept_opt, concept_dict)

    viewpoint_stack = None
    viewpoint_stack_around = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.save_process:
        save_folder_proc = os.path.join(scene.args._model_path, "process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)  # makedirs
        process_view_points = scene.getCircleVideoCameras(
            batch_size=opt.pro_frames_num, render45=opt.pro_render_45
        ).copy()
        save_process_iter = opt.iterations // len(process_view_points)
        pro_img_frames = []

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)
        gaussians.update_rotation_learning_rate(iteration)
        gaussians.update_scaling_learning_rate(iteration)
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # progressively relaxing view range
        if not opt.use_progressive:
            if iteration >= opt.progressive_view_iter and iteration % opt.scale_up_cameras_iter == 0:
                scene.pose_args.fovy_range[0] = max(
                    scene.pose_args.max_fovy_range[0], scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0]
                )
                scene.pose_args.fovy_range[1] = min(
                    scene.pose_args.max_fovy_range[1], scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1]
                )

                scene.pose_args.radius_range[1] = max(
                    scene.pose_args.max_radius_range[1], scene.pose_args.radius_range[1] * opt.scale_up_factor
                )
                scene.pose_args.radius_range[0] = max(
                    scene.pose_args.max_radius_range[0], scene.pose_args.radius_range[0] * opt.scale_up_factor
                )

                scene.pose_args.theta_range[1] = min(
                    scene.pose_args.max_theta_range[1], scene.pose_args.theta_range[1] * opt.phi_scale_up_factor
                )
                scene.pose_args.theta_range[0] = max(
                    scene.pose_args.max_theta_range[0], scene.pose_args.theta_range[0] * 1 / opt.phi_scale_up_factor
                )

                # opt.reset_resnet_iter = max(500, opt.reset_resnet_iter // 1.25)
                scene.pose_args.phi_range[0] = max(
                    scene.pose_args.max_phi_range[0], scene.pose_args.phi_range[0] * opt.phi_scale_up_factor
                )
                scene.pose_args.phi_range[1] = min(
                    scene.pose_args.max_phi_range[1], scene.pose_args.phi_range[1] * opt.phi_scale_up_factor
                )

                print("scale up theta_range to:", scene.pose_args.theta_range)
                print("scale up radius_range to:", scene.pose_args.radius_range)
                print("scale up phi_range to:", scene.pose_args.phi_range)
                print("scale up fovy_range to:", scene.pose_args.fovy_range)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getRandTrainCameras().copy()

        C_batch_size = guidance_opt.C_batch_size
        viewpoint_cams = []
        images = []
        depths = []
        alphas = []
        scales = []
        concept_masks = []

        if concept_opt.concept_type == "CISM":
            text_concept_dict = {}
            text_concept_dict["base"] = []
            text_concept_dict["bg"] = []
            for i in range(concept_opt.concept_num):
                concept_key = f"concept{i}"
                text_concept_dict[concept_key] = []
        elif concept_opt.concept_type == "FEDAVG":
            text_concept_dict = {}
            text_concept_dict["base"] = []
        else:
            pass

        text_z_inverse = torch.cat(
            [
                concept_dict["base"]["bg_text_embedding"]["uncond"],
                concept_dict["base"]["bg_text_embedding"]["inverse_text"],
            ],
            dim=0,
        )

        for i in range(C_batch_size):
            try:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            except:
                viewpoint_stack = scene.getRandTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # pred text_z
            azimuth = viewpoint_cam.delta_azimuth

            # NOTE get view dependent text embeddings here.
            if concept_opt.concept_type == "CISM":
                text_embedding_cism(guidance_opt, concept_opt, concept_dict, azimuth, text_concept_dict)
            elif concept_opt.concept_type == "FEDAVG":
                text_embedding_fedavg(guidance_opt, concept_opt, concept_dict, azimuth, text_concept_dict)
            else:
                NotImplementedError("Not supported")

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                sh_deg_aug_ratio=dataset.sh_deg_aug_ratio,
                bg_aug_ratio=dataset.bg_aug_ratio,
                shs_aug_ratio=dataset.shs_aug_ratio,
                scale_aug_ratio=dataset.scale_aug_ratio,
            )
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]
            concept_mask = render_pkg["render_concept"]

            scales.append(render_pkg["scales"])
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            viewpoint_cams.append(viewpoint_cams)
            concept_masks.append(concept_mask)

        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)
        alphas = torch.stack(alphas, dim=0)

        # Binary mask
        concept_masks = torch.stack(concept_masks, dim=0)
        concept_masks = torch.where(concept_masks > 0.2, torch.tensor(1.0), torch.tensor(0.0))

        if concept_opt.concept_type == "CISM" or concept_opt.concept_type == "FEDAVG":
            # Stacking text embeddings.
            for key in text_concept_dict.keys():
                text_concept_dict[key] = torch.stack(text_concept_dict[key], 1)

        # Loss
        warm_up_rate = 1.0 - min(iteration / opt.warmup_iter, 1.0)

        _aslatent = False
        if iteration < opt.geo_iter or random.random() < opt.as_latent_ratio:
            _aslatent = True

        if concept_opt.concept_type == "CISM":
            loss = guidance.train_step_cism(
                text_concept_dict,
                images,
                concept_masks,
                pred_depth=depths,
                pred_alpha=alphas,
                grad_scale=guidance_opt.lambda_guidance,
                save_folder=save_folder,
                iteration=iteration,
                warm_up_rate=warm_up_rate,
                resolution=(gcams.image_h, gcams.image_w),
                guidance_opt=guidance_opt,
                as_latent=_aslatent,
                embedding_inverse=text_z_inverse,
                concept_opt=concept_opt,
            )
        elif concept_opt.concept_type == "FEDAVG":
            loss = guidance.train_step_fedavg(
                text_concept_dict["base"],
                images,
                pred_depth=depths,
                pred_alpha=alphas,
                grad_scale=guidance_opt.lambda_guidance,
                save_folder=save_folder,
                iteration=iteration,
                warm_up_rate=warm_up_rate,
                resolution=(gcams.image_h, gcams.image_w),
                guidance_opt=guidance_opt,
                as_latent=_aslatent,
                embedding_inverse=text_z_inverse,
            )
        else:
            loss = guidance.train_step(
                text_concept_dict["bg"],
                images,
                pred_depth=depths,
                pred_alpha=alphas,
                grad_scale=guidance_opt.lambda_guidance,
                save_folder=save_folder,
                iteration=iteration,
                warm_up_rate=warm_up_rate,
                resolution=(gcams.image_h, gcams.image_w),
                guidance_opt=guidance_opt,
                as_latent=_aslatent,
                embedding_inverse=text_z_inverse,
            )

        scales = torch.stack(scales, dim=0)
        loss_scale = torch.mean(scales, dim=-1).mean()
        loss_tv = tv_loss(images) + tv_loss(depths)

        loss = (
            loss + opt.lambda_tv * loss_tv + opt.lambda_scale * loss_scale
        )  # opt.lambda_tv * loss_tv + opt.lambda_bin * loss_bin + opt.lambda_scale * loss_scale +
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(viewpoint_cam_p, gaussians, pipe, background, test=True)
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0)
                    img_p = img_p.detach().cpu().permute(1, 2, 0).numpy()
                    img_p = (img_p * 255).round().astype("uint8")
                    pro_img_frames.append(img_p)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                concept_opt.concept_num,
                wandb_logger,
            )
            if iteration in testing_iterations:
                if save_video:
                    video_inference(iteration, scene, render, (pipe, background), wandb_logger)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold
                    )

                if (
                    iteration % opt.opacity_reset_interval == 0
                ):  # or (dataset._white_background and iteration == opt.densify_from_iter)
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene._model_path + "/chkpnt" + str(iteration) + ".pth")

    if opt.save_process:
        imageio.mimwrite(os.path.join(save_folder_proc, "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)

    # Test step saving and Evaluation
    print()
    test_report(
        iteration,
        scene,
        render,
        render_concept,
        (pipe, background),
        concept_dict,
        concept_opt.concept_num,
        wandb_logger,
    )


def prepare_output_and_logger(args):
    if not args._model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args._model_path = os.path.join("./output/", args.workspace)

    # Set up output folder
    print("Output folder: {}".format(args._model_path))
    os.makedirs(args._model_path, exist_ok=True)

    # copy configs
    if args.opt_path is not None:
        os.system(" ".join(["cp", args.opt_path, os.path.join(args._model_path, "config.yaml")]))

    with open(os.path.join(args._model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args._model_path)
    else:
        print("Tensorboard not available: not logging progress")

    return tb_writer


def training_report(
    tb_writer,
    iteration,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    concept_num=1,
    wandb_logger=False,
):
    if tb_writer:
        tb_writer.add_scalar("iter_time", elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        save_folder = os.path.join(scene.args._model_path, "test_six_views/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print("test views is in :", save_folder)
        torch.cuda.empty_cache()
        config = {"name": "test", "cameras": scene.getTestCameras()}

        viewpoint_uids = []
        if config["cameras"] and len(config["cameras"]) > 0:
            for idx, viewpoint in enumerate(config["cameras"]):
                render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
                rgb, depth = render_out["render"], render_out["depth"]
                concept_mask = render_out["render_concept"]
                concept_mask = torch.where(concept_mask > 0.2, torch.tensor(1.0), torch.tensor(0.0))
                if depth is not None:
                    depth_norm = depth / depth.max()
                    save_image(depth_norm, os.path.join(save_folder, "render_depth_{}.png".format(viewpoint.uid)))

                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image, os.path.join(save_folder, "render_view_{}.png".format(viewpoint.uid)))

                for cn in range(concept_num):
                    save_image(
                        concept_mask[cn, None, :, :],
                        os.path.join(save_folder, "render_view_{}_concept_{}.png".format(viewpoint.uid, cn)),
                    )
                viewpoint_uids.append(viewpoint.uid)
                if tb_writer:
                    tb_writer.add_images(
                        config["name"] + "_view_{}/render".format(viewpoint.uid), image[None], global_step=iteration
                    )

            if wandb_logger:
                wandb.log(
                    {
                        "train": [
                            wandb.Image(os.path.join(save_folder, "render_view_{}.png".format(uid)))
                            for uid in viewpoint_uids
                        ]
                    }
                )

            print("\n[ITER {}] Eval Done!".format(iteration))
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


@torch.no_grad()
def test_report(
    iteration,
    scene: Scene,
    renderFunc,
    renderFunc_concept,
    renderArgs,
    concept_dict,
    concept_num=1,
    wandb_logger=False,
):

    # Report test and samples of training set
    save_folder = os.path.join(scene.args._model_path, "test_save/{}_iteration".format(iteration))
    concept_dict["base"]["render_save_path"] = save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("test views is in :", save_folder)

    for i in range(concept_num):
        concept_key = f"concept{i}"
        save_folder_concept = os.path.join("{}_concept{}".format(save_folder, i))
        concept_dict[concept_key]["render_save_path"] = save_folder_concept
        if not os.path.exists(save_folder_concept):
            os.makedirs(save_folder_concept)
            print(f"test views concept {i} is in :", save_folder_concept)

    torch.cuda.empty_cache()
    config = {"name": "test", "cameras": scene.getCircleVideoCamerasPolars(batch_size=120, polars=70)}
    if config["cameras"] and len(config["cameras"]) > 0:
        for idx, viewpoint in enumerate(config["cameras"]):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb, depth = render_out["render"], render_out["depth"]
            concept_mask = render_out["render_concept"]
            concept_mask = torch.where(concept_mask > 0.2, torch.tensor(1.0), torch.tensor(0.0))

            image = torch.clamp(rgb, 0.0, 1.0)
            save_image(image, os.path.join(save_folder, "{}.png".format(viewpoint.uid)))

            for i in range(concept_num):
                render_out = renderFunc_concept(viewpoint, scene.gaussians, *renderArgs, concept_num=i)
                rgb, depth = render_out["render"], render_out["depth"]
                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image, os.path.join("{}_concept{}".format(save_folder, i), "{}.png".format(viewpoint.uid)))

            if wandb_logger:
                wandb.log({"test": wandb.Image(os.path.join(save_folder, "{}.png".format(viewpoint.uid)))})

        print("\n[ITER {}] Eval Done!".format(iteration))
    torch.cuda.empty_cache()


def video_inference(iteration, scene: Scene, renderFunc, renderArgs, wandb_logger):
    sharp = T.RandomAdjustSharpness(3, p=1.0)

    save_folder = os.path.join(scene.args._model_path, "videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs
        print("videos is in :", save_folder)
    torch.cuda.empty_cache()
    config = {"name": "test", "cameras": scene.getCircleVideoCameras()}
    if config["cameras"] and len(config["cameras"]) > 0:
        img_frames = []
        depth_frames = []
        print("Generating Video using", len(config["cameras"]), "different view points")
        for idx, viewpoint in enumerate(config["cameras"]):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb, depth = render_out["render"], render_out["depth"]
            if depth is not None:
                depth_norm = depth / depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0)
                depths = depths.detach().cpu().permute(1, 2, 0).numpy()
                depths = (depths * 255).round().astype("uint8")
                depth_frames.append(depths)

            image = torch.clamp(rgb, 0.0, 1.0)
            image = image.detach().cpu().permute(1, 2, 0).numpy()
            image = (image * 255).round().astype("uint8")
            img_frames.append(image)
            # save_image(image,os.path.join(save_folder,"lora_view_{}.jpg".format(viewpoint.uid)))
        # Img to Numpy
        imageio.mimwrite(
            os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, quality=8
        )
        if len(depth_frames) > 0:
            imageio.mimwrite(
                os.path.join(save_folder, "video_depth_{}.mp4".format(iteration)), depth_frames, fps=30, quality=8
            )
        print("\n[ITER {}] Video Save Done!".format(iteration))
        if wandb_logger:
            wandb.log({"video": wandb.Video(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), fps=4)})
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import yaml

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument("--opt", type=str, default=None)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_ratio", type=int, default=5)  # [2500,5000,7500,10000,12000]
    parser.add_argument("--save_ratio", type=int, default=2)  # [10000,12000]
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--save_suffix", type=str, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project_name", type=str, default="ISM")
    parser.add_argument("--wandb_name", type=str, default="ISM")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gcp = GenerateCamParams(parser)
    gp = GuidanceParams(parser)
    cp = ConceptParams(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        lp.load_yaml(opts.get("ModelParams", None))
        op.load_yaml(opts.get("OptimizationParams", None))
        pp.load_yaml(opts.get("PipelineParams", None))
        gcp.load_yaml(opts.get("GenerateCamParams", None))
        gp.load_yaml(opts.get("GuidanceParams", None))
        cp.load_yaml(opts.get("ConceptParams", None))

        lp.opt_path = args.opt
        args.port = opts["port"]
        args.save_video = opts.get("save_video", True)
        args.seed = opts.get("seed", 0)
        args.device = opts.get("device", "cuda")

        # override device
        gp.g_device = args.device
        lp.data_device = args.device
        gcp.device = args.device

        # Saving directory
        lp.workspace = os.path.join(args.save_suffix, lp.workspace)
        cp.gpt_key = GPT_API_KEY

    args.wandb = False
    if args.wandb:
        wandb.init(project=args.wandb_project_name)
        wandb.run.name = args.wandb_name

    # save iterations
    test_iter = [1] + [k * op.iterations // args.test_ratio for k in range(1, args.test_ratio)] + [op.iterations]
    args.test_iterations = test_iter

    save_iter = [k * op.iterations // args.save_ratio for k in range(1, args.save_ratio)] + [op.iterations]
    args.save_iterations = save_iter

    print("Test iter:", args.test_iterations)
    print("Save iter:", args.save_iterations)
    print("Optimizing " + lp._model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp,
        op,
        pp,
        gcp,
        gp,
        cp,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.save_video,
        args.wandb,
    )

    # All done
    print("\nTraining complete.")
