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

import json
import os
import random

from arguments import ConceptParams, GenerateCamParams, ModelParams
from scene.dataset_readers import (
    GenerateCircleCameras,
    GenerateCircleCamerasPolars,
    GeneratePurnCameras,
    GenerateRandomCameras,
    sceneLoadTypeCallbacks,
)
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos, cameraList_from_RcamInfos
from utils.system_utils import searchForMaxIteration


class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        pose_args: GenerateCamParams,
        concept_args: ConceptParams,
        gaussians: GaussianModel,
        concept_dict: dict,
        load_iteration=None,
        shuffle=False,
        resolution_scales=[1.0],
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args._model_path
        self.pretrained_model_path = args.pretrained_model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scales = resolution_scales
        self.pose_args = pose_args
        self.args = args
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}
        scene_info = sceneLoadTypeCallbacks["RandomCamConcept"](self.model_path, pose_args, concept_args, concept_dict)

        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = pose_args.default_radius  #    scene_info.nerf_normalization["radius"]
        for resolution_scale in resolution_scales:
            self.test_cameras[resolution_scale] = cameraList_from_RcamInfos(
                scene_info.test_cameras, resolution_scale, self.pose_args
            )

        # FIX가 되어야할듯

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
            )
        elif self.pretrained_model_path is not None:
            self.gaussians.load_ply(self.pretrained_model_path)
        else:
            self.gaussians.create_from_multiple_pcd_scene(concept_dict, concept_args, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getRandTrainCameras(self, scale=1.0):
        rand_train_cameras = GenerateRandomCameras(self.pose_args, self.args.batch, SSAA=True)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(
                rand_train_cameras, resolution_scale, self.pose_args, SSAA=True
            )
        return train_cameras[scale]

    def getPurnTrainCameras(self, scale=1.0):
        rand_train_cameras = GeneratePurnCameras(self.pose_args)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(
                rand_train_cameras, resolution_scale, self.pose_args
            )
        return train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getCircleVideoCameras(self, scale=1.0, batch_size=120, render45=True):
        video_circle_cameras = GenerateCircleCameras(self.pose_args, batch_size, render45)
        video_cameras = {}
        for resolution_scale in self.resolution_scales:
            video_cameras[resolution_scale] = cameraList_from_RcamInfos(
                video_circle_cameras, resolution_scale, self.pose_args
            )
        return video_cameras[scale]

    def getCircleVideoCamerasPolars(self, scale=1.0, batch_size=120, polars=60):
        video_circle_cameras = GenerateCircleCamerasPolars(self.pose_args, batch_size, polars)
        video_cameras = {}
        for resolution_scale in self.resolution_scales:
            video_cameras[resolution_scale] = cameraList_from_RcamInfos(
                video_circle_cameras, resolution_scale, self.pose_args
            )
        return video_cameras[scale]

    def update_gaussian_with_3d_layouts(self, concept_dict, concept_args):
        self.gaussians.create_from_multiple_pcd_scene(concept_dict, concept_args, self.cameras_extent)
        print("Gaussian updated with refined transformations")
