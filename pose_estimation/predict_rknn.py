import os
import sys
import time
from functools import wraps
import numpy as np
import cv2
from rknn.api import RKNN
import torch

from zalaiConvert.utils.cameraViewer import CameraViewer  
from zalaiConvert.utils.farward_utils import activateEnv, predictWrap, draw_pts, timeit
from zalaiConvert.utils.rknn_utils import getRknn
from zalaiConvert.utils.keypoint_utils import get_max_preds
from _predictor import timeit, draw_pts

activateEnv()


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
    parser.add_argument('--input-size', nargs='*', type=int, default=[256,256], help='input image size(H, W): 256 256')

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

class RknnPredictor():
    def __init__(self, rknn, input_size=None):
        super(RknnPredictor, self).__init__()
        self.rknn = rknn

        self.width = 256
        self.height = 256
        if input_size:
            self.height, self.width = tuple(input_size)
        print(self.height, self.width)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)*255
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)*255
        self.output_size = [self.height // 4, self.width//4]

    def preprocess(self, img, with_normalize=None, hwc_chw=None, **kwargs):
        if img.shape[0:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))

        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32)

        # if with_normalize:
        #     input_image = (input_image - self.mean) / self.std
        return [input_image]

    def farward(self, x):
        outputs = self.rknn.inference(inputs=x)
        return outputs[0]

    def postProcess(self, score_map):
        if not isinstance(score_map, torch.Tensor):
            print("trans to tensor")
            score_map = torch.Tensor(score_map)

        # print(score_map.shape)
        kpts, _ = get_max_preds(score_map.numpy())
        kpts = kpts.squeeze(0) * 4
        return kpts

    @timeit
    def predict(self, x, args=None):
        input_tensor = self.preprocess(x)
        score_map = self.farward(input_tensor)
        kpts = self.postProcess(score_map)
        return kpts

    def draw(self, img, preds):
        print(preds)
        return draw_pts(img, preds)

    # def __del__(self):
    #     self.rknn.release()



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

    model = RknnPredictor(rknn, input_size=args.input_size)
    # model = RknnPredictor(1)
    predictWrap(args.input, model, args)
    print("__________________exit__________________")

if __name__ == "__main__":
    cmds = ['weights/coco_256x192_q.rknn', '--device', 'rk1808', '-i',  "data/abc*.jpg", '--show-img']
    # cmds = ['weights/tmp/pose_resnet_50_256x256_q.rknn', '--device', 'rk1808', '-i',  "data/abc*.jpg", '--show-img']
    cmds = ['weights/face300b_256x256_q.rknn', '--device', 'rk1808', '-i',  "data/faces/*.png", '--show-img']
    main(cmds)
