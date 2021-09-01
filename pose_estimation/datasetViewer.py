import argparse
import os
import pprint
import shutil

import numpy as np
import torch
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
import dataset
import models
import cv2

from utils.transforms import transform_preds
from core.inference import get_max_preds
from core.inference import get_final_preds

"""
    dataset.coco

    @returns 
    input: shape=[3 256 192]
    target: shape=[17,64,48]
    target_weight: shape=[17,1]
    meta: list(len=9) 
"""   
COCO_ROOT = r"H:\Dataset\keypoint\coco2017" 

def main():
    cfg_file = r"experiments\coco\resnet50\256x192_d256x3_adam_lr1e-3_caffe.yaml"
    update_config(cfg_file)
    config.TEST.POST_PROCESS = False
    demo()

def demo2():
# Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = dataset.coco(
        cfg=config,
        root=COCO_ROOT,
        image_set="val2017",
        is_train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    for i in range(0,15):
        img, tg, tgw,meta = train_dataset[i]
        # print(img)
        # img = img.astype(np.uint8)
        tg = tg.unsqueeze(0)
        # coords, maxvals = get_max_preds(tg.numpy())
        coords, maxvals = get_final_preds(config, tg.numpy(), meta["center"], meta["scale"])
        print(coords, meta["center"], meta["scale"])
        img2 = cv2.imread(meta["image"])
        # print(img2)
        img2 = draw_pts(img2, coords.squeeze(0))
        cv2.imshow('aa', img2)
        cv2.waitKey(0)

def demo():
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = dataset.coco(
        cfg=config,
        root=COCO_ROOT,
        image_set="val2017",
        is_train=False,
        # transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize,
        # ])
    )
    for i in range(0,15,5):
        img, tg, tgw,meta = train_dataset[i]
        # print(img)
        img = img.astype(np.uint8)
        tg = tg.unsqueeze(0)
        coords, maxvals = get_max_preds(tg.numpy())
        print(coords, meta["image"], meta["center"], meta["scale"])
        cv2.imwrite('abc.jpg', img)
        img2 = draw_pts(img, coords.squeeze(0)*4)
        cv2.imshow('aa', img2)
        print(img2.shape)
        
        cv2.waitKey(0)


def draw_pts(img, kpts):
    img2 = img.copy()
    for k in kpts:
        x = int(k[0])
        y = int(k[1])
        cv2.circle(img2, (x, y), radius=2, thickness=-1, color=(0, 0, 255))
    return img2

if __name__ == "__main__":
    main()
