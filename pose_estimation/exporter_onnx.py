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
import numpy as np
import cv2

import _init_paths
from core.config import config
from core.config import update_config
from utils.utils import create_logger
from cameraViewer import CameraViewer

def export(model, output, output_type='onnx', shape=(1,3,256,192)):
    from torch.autograd import Variable

    model = model.eval()

    x = Variable(torch.randn(*shape), requires_grad=False)
    with torch.no_grad():
        if output_type == 'onnx':
            torch.onnx.export(model, x, output, verbose=True, training=False,
                do_constant_folding=True, opset_version=7)
        elif output_type == 'torchscript':
            x = x.float()
            assert isinstance(model, torch.nn.Module)
            traced_script_module = torch.jit.trace(model, x)
            # traced_script_module = torch.jit.trace_module(model, x)
            traced_script_module.save(output)
        else:
            print("not found", output_type)

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
    parser.add_argument('--shape-list', type=int, nargs='*', 
                        help='shape list, such as 1 3 256 192')
    parser.add_argument('--output', '-o', default='out.onnx',
                        help='output image path')                   

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
    
def modelFactory(model_path, config):
    from models.pose_resnet import get_pose_net
    model = get_pose_net(config, is_train=False)

    ckpt = torch.load(model_path)

    model_dict = {}
    for k,v in ckpt.items():
        if "module" in k:
            k = k[7:]# .lstrip('module\\.')
        model_dict[k] = v
        
    model.load_state_dict(model_dict)
    return model

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

    model = modelFactory(config.TEST.MODEL_FILE, config)

    export(model, args.output, shape=args.shape_list)
    # export(model, 'abc.torchscript', 'torchscript')


if __name__ == '__main__':
    # cmds = ['--cfg', r'experiments\mpii\resnet50\256x256_d256x3_adam_lr1e-3.yaml', 
    # '--model-file', r'models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar']
    cmds = ['--cfg', r'experiments\coco\resnet50\256x192_d256x3_adam_lr1e-3.yaml', 
    '--model-file', r'output\coco\pose_resnet_50\256x192_d256x3_adam_lr1e-3\model_best.pth.tar',
    '-o', 'weights/coco_256x192.onnx']

    cmds = ['--cfg', r'experiments\face300w\256x256_d256x3_adam_lr1e-3_a.yaml', 
    '--model-file', r'output\CsvKptDataset\pose_resnet_50\256x256_d256x3_adam_lr1e-3_a\model_best.pth.tar',
    '--shape-list', '1', '3', '256', '256', '-o', 'weights/face300_256x256f.onnx']
    cmds = ['--cfg', r'experiments\face300w\256x256_d256x3_adam_lr1e-3_a.yaml', 
    '--model-file', r'output\CsvKptDataset\pose_resnet_50\256x256_d256x3_adam_lr1e-3_b\model_best.pth.tar',
    '--shape-list', '1', '3', '256', '256', '-o', 'weights/face300b_256x256.onnx']
    
    main()
