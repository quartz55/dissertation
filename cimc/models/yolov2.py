import os
import urllib.parse

import time
from typing import List

import imageio
from PIL import Image
import numpy as np
import lightnet as ln
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from .labels import COCO_LABELS

YOLOV2_VOC_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov2.weights"
DIMENSION = [416, 416]


class YoloV2(ln.models.Yolo):
    def detect_image(self, image):
        if isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, str):
            img = imageio.imread(image)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError(f"image must be of type Image, str or ndarray")

        height, width = img.shape[:2]

        t_prepare = time.time()
        pre_process = transforms.Compose([
            ln.data.Letterbox(dimension=DIMENSION),
            transforms.ToTensor()
        ])
        tensor = pre_process(img).unsqueeze(0).cuda()
        input_img = Variable(tensor, volatile=True)
        t_prepare = time.time() - t_prepare

        t_detect = time.time()
        boxes = self(input_img)
        boxes = ln.data.ReverseLetterbox.apply(boxes, DIMENSION, (width, height))
        t_detect = time.time() - t_detect

        timings = {
            'prepare': t_prepare,
            'detect': t_detect
        }
        return boxes, img, timings

    @classmethod
    def pre_trained(cls, weights: str, labels: List[str] = COCO_LABELS,
                    confidence: float = 0.25, nms: float = 0.4):
        assert weights is not None, "Please provide weights"
        net = cls(num_classes=len(labels), weights_file=weights, conf_thresh=confidence, nms_thresh=nms)
        net.postprocess = transforms.Compose([
            net.postprocess,
            ln.data.TensorToBrambox(network_size=DIMENSION,
                                    class_label_map=None)
        ])
        net.eval()
        return net
