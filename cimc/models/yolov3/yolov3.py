from attr._make import _CountingAttr

from . import darknet as d
from cimc.core.vec import Vec2

import os
import time
from PIL import Image
from typing import Union, NamedTuple, List, Optional, Tuple
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import attr

COCO_LABELS = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]

YOLOV3_CFG = os.path.join(os.path.dirname(__file__), 'yolov3.cfg')

ImageType = Union[str, np.ndarray, Image.Image]


class YoloV3(d.Darknet):
    def __init__(self):
        super().__init__(YOLOV3_CFG)

    def detect_image(self, image: ImageType, confidence=0.25, nms_thres=0.4):
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image, 'RGB')
        else:
            raise TypeError(f"image must be of type {ImageType}")

        pre_process = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor()
        ])

        if self.training:
            self.eval()
        t0 = time.time()

        img_input = pre_process(img).unsqueeze(0)
        t1 = time.time()

        if next(self.parameters()).is_cuda:
            img_input = img_input.cuda()
        img_input = Variable(img_input, volatile=True)
        t2 = time.time()

        boxes_list = self(img_input)
        boxes = boxes_list[0][0] + boxes_list[1][0] + boxes_list[2][0]
        t3 = time.time()

        boxes = nms(boxes, nms_thres)
        t4 = time.time()

        timings = {
            'pre_process': t1 - t0,
            'cuda': t2 - t1,
            'predict': t3 - t2,
            'nms': t4 - t3,
            'total': t4 - t0
        }
        return boxes, img, timings

    @classmethod
    def pre_trained(cls, weights: str):
        net = cls()
        net.load_weights(weights)
        return net


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    confidences = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        confidences[i] = 1 - boxes[i][4]

    _, sort_ids = torch.sort(confidences)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sort_ids[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sort_ids[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        min_x = min(box1[0], box2[0])
        max_x = max(box1[2], box2[2])
        min_y = min(box1[1], box2[1])
        max_y = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        min_x = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        max_x = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        min_y = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        max_y = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    u_width = max_x - min_x
    u_height = max_y - min_y
    c_width = w1 + w2 - u_width
    c_height = h1 + h2 - u_height
    if c_width <= 0 or c_height <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    c_area = c_width * c_height
    u_area = area1 + area2 - c_area
    return c_area / u_area
