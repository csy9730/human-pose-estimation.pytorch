# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim

from core.config import get_model_name

def get_output_dir(cfg, cfg_name, output=None):
    output_dir, output_name = None, os.path.basename(cfg_name).split('.')[0]
    if output:
        output_dir, output_name = os.path.split(output)
        output_name = os.path.basename(output_name).split('.')[0]

    if output_dir:
        final_output_dir = output_dir
    else:
        root_output_dir = Path(cfg.OUTPUT_DIR)
        # set up logger
        if not root_output_dir.exists():
            print('=> creating {}'.format(root_output_dir))
            root_output_dir.mkdir()
        dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
            if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
        dataset = dataset.replace(':', '_')
        model, _ = get_model_name(cfg)
        final_output_dir = root_output_dir / dataset / model / output_name   

    print('=> creating {}'.format(final_output_dir))
    os.makedirs(final_output_dir, exist_ok=True)
    return str(final_output_dir), output_name

def create_log(final_log_file):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger

def create_logger(cfg, cfg_name, phase='train', output=None):
    final_output_dir, name = get_output_dir(cfg, cfg_name, output=output)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(name, time_str, phase)
    final_log_file = os.path.join(final_output_dir, log_file)
    logger = create_log(final_log_file)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / (name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, final_output_dir, str(tensorboard_log_dir)

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))
