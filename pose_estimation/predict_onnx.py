# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse
import cv2
import math
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
from _predictor import PosenetPredictor, predictWrap, timeit
__DIR__ = os.path.dirname(os.path.abspath(__file__))


class PosenetOnnxPredictor(PosenetPredictor):
    def __init__(self, model_file):
        super(PosenetOnnxPredictor, self).__init__()
        self.loadModel(model_file)
        # self.width = 256
        # self.height = 256
        
    def loadModel(self, f):
        import onnxruntime as rt
        sess = rt.InferenceSession(f)
        self.input_name = sess.get_inputs()[0].name
        self.label_name = sess.get_outputs()[0].name
        print('input_name: ' + self.input_name)
        print('label_name: ' + self.label_name)
        self.sess = sess
        return self.sess

    def farward(self, x):
        pred_onx = self.sess.run([self.label_name], {self.input_name: x})
        return pred_onx[0]

    @timeit
    def predict(self, img):
        input_tensor = self.preprocess(img)
        score_map = self.farward(input_tensor.numpy())
        kpts = self.postProcess(torch.from_numpy(score_map))
        
        return kpts


def parse_args(cmds=None):
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--model-in', '-m', dest="model_file", help='model parameters', required=True)
    parser.add_argument('--input','-i',help='input image path')
    args = parser.parse_args(cmds)
    return args


def main(cmds=None):
    args = parse_args(cmds)

    model = PosenetOnnxPredictor(args.model_file)
    predictWrap(args.input, model, args)


if __name__ == '__main__':
    cmds = ["--model-in", "weights/coco_256x192.onnx"]
    # cmds = ["--model-in", "pose_resnet_50_256x256/pose_resnet_50_256x256.onnx"]
    cmds += ['-i', "data/abc*.jpg"]
    main(cmds)
