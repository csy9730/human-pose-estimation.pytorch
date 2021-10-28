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
from _predictor import predictWrap, timeit, draw_pts
__DIR__ = os.path.dirname(os.path.abspath(__file__))


def parse_args(cmds=None):
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--model-in', '-m', dest="model_file", help='model parameters', required=True)
    parser.add_argument('--input','-i',help='input image path')
    parser.add_argument('--input-size', nargs='*', type=int, help='input image size(H, W): 256 192')
    args = parser.parse_args(cmds)
    return args

class PosenetOnnxPredictor():
    def __init__(self, model_file):
        super(PosenetOnnxPredictor, self).__init__()
        self.loadModel(model_file)

        self.width = 256 # 256
        self.height = 256
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.output_size = [self.height // 4, self.width//4]
    
    def loadModel(self, f):
        import onnxruntime as rt
        sess = rt.InferenceSession(f)
        self.input_name = sess.get_inputs()[0].name
        self.label_name = sess.get_outputs()[0].name
        print('input_name: ' + self.input_name)
        print('label_name: ' + self.label_name)
        self.sess = sess
        return self.sess

    def preprocess(self, img):
        if img.shape[0:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))

        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #print("input_image.shape:", input_image.shape)
        input_image = input_image.astype(np.float32)
        input_image = (input_image - self.mean) / self.std
        input_image = input_image.transpose([2, 0, 1])

        input_tensor = torch.tensor(input_image)
        input_tensor = input_tensor.unsqueeze(0)
        #print("input_tensor.shape", input_tensor.shape)
        return input_tensor

    def farward(self, x):
        pred_onx = self.sess.run([self.label_name], {self.input_name: x})
        return pred_onx[0]

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
    def predict(self, img):
        input_tensor = self.preprocess(img)
        score_map = self.farward(input_tensor.numpy())
        kpts = self.postProcess(torch.from_numpy(score_map))
        
        return kpts

    def draw(self, img, preds):
        return draw_pts(img, preds)



def main(cmds=None):
    args = parse_args(cmds)

    model = PosenetOnnxPredictor(args.model_file)
    if args.input_size:
        model.height = args.input_size[0]
        model.width = args.input_size[1]

    predictWrap(args.input, model, args)


if __name__ == '__main__':
    cmds = ["--model-in", "weights/coco_256x192.onnx", '-i', "data/abc*.jpg", '--input-size', '256', '192']
    # cmds = ["--model-in", "weights\pose_resnet_50_256x256/pose_resnet_50_256x256.onnx", '-i', "data/abc*.jpg"]

    cmds = ["--model-in", r'weights/face300_256x256.onnx', '-i', 'data/faces/*.png']
    main()
