# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import time

import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np

import _init_paths
from core.config import config
from core.config import update_config
from utils.utils import create_logger
import dataset
import models
from _predictor import PosenetPredictor, predictWrap


def parse_args(cmds=None):
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name')
    
    args, rest = parser.parse_known_args(cmds)
    # args.cfg = r"experiments\coco\resnet50\256x192_d256x3_adam_lr1e-3_caffe.yaml"
    # update config
    print(args.cfg,'cfg')
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args(cmds)

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def main(cmds=None):
    args = parse_args(cmds)
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        ckpt = torch.load(config.TEST.MODEL_FILE)

        model_dict = {}
        for k,v in ckpt.items():
            if "module" in k:
                k = k[7:]# .lstrip('module\\.')
            model_dict[k] = v
            
        model.load_state_dict(model_dict)

    # export(model, 'coco_256x192.onnx')
    # exit(0)
    # export(model, 'abc.torchscript', 'torchscript')

    pdter = PosenetPredictor()
    pdter.model = model
    
    imgpath = r"H:\Dataset\keypoint\lsp\lsp_dataset\images\*.jpg"
    # imgpath = r"data/abc*.jpg"
    predictWrap(imgpath, pdter)


if __name__ == '__main__':
    # cmds = ['--cfg', r'experiments\mpii\resnet50\256x256_d256x3_adam_lr1e-3.yaml', 
    # '--model-file', r'models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar']
    cmds = ['--cfg', r'experiments\coco\resnet50\256x192_d256x3_adam_lr1e-3.yaml', 
    '--model-file', r'output\coco\pose_resnet_50\256x192_d256x3_adam_lr1e-3\model_best.pth.tar']
    
    main(cmds)
