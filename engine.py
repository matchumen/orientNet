import math
import sys
import time
import torch
import numpy as np

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import pprint


def fit(model, optimizer, data_loader, device, epochs, lr_scheduler):
    model.train()
    for epoch in range(epochs):
        for images, targets in data_loader:
            # move sample to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # backward
            optimizer.zero_grad()
            losses.backward()

            optimizer.step()
            lr_scheduler.step()

            out = ""
            for k, v in loss_dict.items():
                out += str(k) + " " + str('{:.5f}'.format(v.item())) + " "
            print(out)

        # print log
        print("#" * 20)
        out = ""
        for k, v in loss_dict.items():
            out += str(k) + " " + str('{:.5f}'.format(v.item())) + " "
        print(out)
        print("#" * 20)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


