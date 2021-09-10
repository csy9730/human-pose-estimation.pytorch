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