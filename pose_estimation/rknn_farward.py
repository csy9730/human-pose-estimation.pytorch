import os
import sys
import time
from functools import wraps
import numpy as np
import cv2
from rknn.api import RKNN
import torch
from zalaiConvert.farward.cameraViewer import CameraViewer  
from zalaiConvert.farward.farward_utils import activateEnv
from zalaiConvert.farward.farward_utils import getRknn
import _init_paths
from _predictor import PosenetPredictor, predictWrap
activateEnv()


class RknnPredictor(PosenetPredictor):
    def __init__(self, rknn):
        super(RknnPredictor, self).__init__()
        self.rknn = rknn
        
    def preprocess(self, img, with_normalize=None, hwc_chw=None, **kwargs):
        if img.shape[0:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))

        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32)

        return [input_image]

    def farward(self, x):
        outputs = self.rknn.inference(inputs=x)
        return outputs[0]

    # @decorator_retval([])  

    # @timeit  
    # def predict(self, img, args):
    #     img2 = self.preprocess(img)
    #     score_map = self.farward(img2)
    #     kpts = self.postProcess(score_map)
    #     return kpts
    
    # def __del__(self):
    #     self.rknn.release()


def parse_args(cmds=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rknn predict & show key points')
    parser.add_argument('model', help='model file path')
    parser.add_argument('--input', '-i', help='image file path')
    parser.add_argument('--output', '-o', help='save output image name')
    parser.add_argument('--config')

    parser.add_argument('--use-padding', action='store_true', help='model file path')
    parser.add_argument('--input-chw', action='store_true', help='model file path')
    parser.add_argument('--with-normalize', action='store_true', help='rknn with normalize')
    parser.add_argument('--hwc-chw', action='store_true', help='image preprocess: from HWC to CHW')

    # parser.add_argument('--target', choices=['rk1808', 'rv1126'], help='target device: rk1808, rk1126')
    parser.add_argument('--device', choices=['rk1808', 'rv1126'], help='device: rk1808, rv1126')
    parser.add_argument('--device-id')
    parser.add_argument('--task', choices=['segment', 'detect', 'classify', 'keypoint'], default='keypoint', help='device: rk1808, rk1126')
    parser.add_argument('--run-perf', action='store_true', help='eval perf')

    parser.add_argument('--verbose', action='store_true', help='verbose information')
    parser.add_argument('--save-npy', action='store_true')
    parser.add_argument('--save-img', action='store_true', help='save image')
    parser.add_argument('--show-img', action='store_true', help='show image')
    parser.add_argument('--mix-scale', type=float, help='segment task params: mix scale')
    return parser.parse_args(cmds)


def predictWrap2(source, model, args):
    cmv = CameraViewer(source)
    imgs = cmv.stream()
    H, W = model.height, model.width
    for i, img in enumerate(imgs):
        if img.shape[0:2] != (H, W):
            img = cv2.resize(img, (W, H))
        kpts = model.predict(img)
        if args.save_npy:
            np.save('out_{0}.npy'.format(i=i), pred[0])
        
        img2 = model.draw(img, kpts)

        # print(kpts, kpts.shape)
        # if args.save_img:
        #     cv2.imwrite(args.output.format(i=i), img2.astype(np.uint8))

        if cmv.use_camera or args.show_img:
            cv2.imshow(cmv.title.format(i=0), img2)
            k = cv2.waitKey(cmv.waitTime)
            if k == 27:
                break
    print("predict finished")

def main(cmds=None):
    args = parse_args(cmds)

    if args.output:
        args.save_img = True
    elif args.output is None and args.save_img:
        args.output = 'out.jpg'

    rknn = getRknn(args.model, device=args.device, device_id=args.device_id)
    if rknn is None:
        exit(-1)

    model = RknnPredictor(rknn)
    # model = RknnPredictor(1)
    predictWrap(args.input, model, args)
    print("__________________exit__________________")

if __name__ == "__main__":
    cmds = ['weights/coco_256x192_q.rknn', '--device', 'rk1808', '-i',  "data/abc*.jpg", '--show-img']
    main(cmds)
