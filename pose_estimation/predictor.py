# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------



import argparse
import os
import pprint
import time

import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np

import _init_paths
from lib.core.config import config
from lib.core.config import update_config
from lib.utils.utils import create_logger
import models
from _predictor import predictWrap, timeit, draw_pts
import torchvision.transforms as transforms

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
    parser.add_argument('--input', '-i')
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


class PosresPredictor(object):
    def __init__(self):
        self.width = 256 # 192
        self.height = 256
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.output_size = [self.height // 4, self.width//4]

    def preprocess(self, img):
        if img.shape[0:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))

        # input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = img

        #print("input_image.shape:", input_image.shape)
        input_image = input_image.astype(np.float32)

        input_image = (input_image /1.0 - self.mean) / self.std
        input_image = input_image.transpose((2, 0, 1))
        input_tensor = torch.from_numpy(input_image)
        
        # normalize = transforms.Normalize(mean=self.mean, std=self.std)
        # tranf = transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        # input_tensor = tranf(input_image)

        input_tensor = input_tensor.unsqueeze(0)
        #print("input_tensor.shape", input_tensor.shape)
        return input_tensor

    def farward(self, x):
        with torch.no_grad():
            ret = self.model(x)
            # print(ret.shape, "ret shape")
            return ret.data.cpu()

    def postProcess(self, score_map):
        if not isinstance(score_map, torch.Tensor):
            print("trans to tensor")
            score_map = torch.Tensor(score_map)
        from core.inference import get_max_preds
        # print(score_map.shape)
        kpts, _ = get_max_preds(score_map.numpy())
        kpts = kpts.squeeze(0) * 4
        return kpts

    @timeit
    def predict(self, x):
        input_tensor = self.preprocess(x)
        score_map = self.farward(input_tensor)
        kpts = self.postProcess(score_map)
        return kpts

    def draw(self, img, preds):
        return draw_pts(img, preds)

def modelFactory(model_path, config):
    from lib.models.pose_resnet import get_pose_net
    model = get_pose_net(config, is_train=False)
    if model_path:
        # logger.info('=> loading model from {}'.format(model_path))
        ckpt = torch.load(model_path)
        if "checkpoint.pth.tar" in model_path:
            ckpt = ckpt["state_dict"]

        model_dict = {}
        for k,v in ckpt.items():
            if "module" in k:
                k = k[7:]# .lstrip('module\\.')
            if k in ["epoch", "model", "state_dict", "perf", "optimizer"]:
                continue
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
    pdter = PosresPredictor()
    model.eval()
    pdter.model = model

    # from exporter import export
    # export(model, 'a.onnx', shape=[1,3,256,256])
    # return
    imgpath = args.input
    # imgpath = r"data/abc*.jpg"
    predictWrap(imgpath, pdter)


if __name__ == '__main__':
    cmds = ['--cfg', r'experiments\face300w\256x256_d256x3_adam_lr1e-3_c.yaml', 
    '--model-file', r'output\CsvKptDataset\pose_resnet_50\256x256_d256x3_adam_lr1e-3_c\checkpoint.pth.tar' , '-i', r"data/faces/*g"]

    main(cmds)
