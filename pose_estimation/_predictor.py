import os
import sys
import time

import numpy as np
import cv2
import torch
from functools import wraps

import _init_paths
from cameraViewer import CameraViewer

def decorator_retval(retval):
    def set_args(f):
        def decorated(*args, **kwargs):
            return retval
        return decorated
    return set_args


def timeit(func): 
    @wraps(func)
    def wrapper(*dargs, **kwargs):
        tic = time.time()
        retval = func(*dargs, **kwargs)
        toc = time.time()
        # time.process_time()
        print('%s() used: %fs' % (func.__name__, toc - tic)) 
        return retval 
    return wrapper


class PosenetPredictor(object):
    def __init__(self):
        # self.loadModel(model_file)
        
        self.width = 192
        self.height = 256
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)*255
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)*255
        self.output_size = [self.height // 4, self.width//4]

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

    def postProcess(self, score_map):
        if not isinstance(score_map, torch.Tensor):
            print("trans to tensor")
            score_map = torch.Tensor(score_map)
        from core.inference import get_max_preds
        # print(score_map.shape)
        kpts, _ = get_max_preds(score_map.numpy())
        kpts = kpts.squeeze(0) * 4
        return kpts

    def farward(self, x):
        with torch.no_grad():
            ret = self.model(x)
            # print(ret.shape, "ret shape")
            return ret.data.cpu()
    
    @timeit
    def predict(self, x):
        input_tensor = self.preprocess(x)
        score_map = self.farward(input_tensor)
        kpts = self.postProcess(score_map)
        return kpts

    def draw(self, img, preds):
        return draw_pts(img, preds)


def draw_pts(img, kpts):
    img2 = img.copy()
    for k in kpts:
        x = int(k[0])
        y = int(k[1])
        cv2.circle(img2, (x, y), radius=2, thickness=-1, color=(0, 0, 255))
    return img2


def predictWrap(source, model, args=None):
    cmv = CameraViewer(source)
    imgs = cmv.stream()
    H, W = model.height, model.width
    for i, img in enumerate(imgs):
        if img.shape[0:2] != (H, W):
            img = cv2.resize(img, (W, H)) 
        kpts = model.predict(img)
        # print(kpts.shape)
        img2 = model.draw(img, kpts)
        cv2.imshow(cmv.title.format(i=i), img2)
        k = cv2.waitKey(cmv.waitTime)
        if k == 27:
            break
    print("predict finished")