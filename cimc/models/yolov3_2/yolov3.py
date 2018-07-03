import os
import time

import torch
from torchvision.transforms import transforms as tf

import cimc.utils as utils
from cimc import resources
from cimc.utils import bench
from .models import Darknet
from .utils.utils import non_max_suppression

YOLOV3_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
YOLOV3_CFG = os.path.join(os.path.dirname(__file__), "yolov3.cfg")

_bench = bench.Bench("yolov3")


class YoloV3(Darknet):
    def __init__(self):
        super().__init__(YOLOV3_CFG)
        self.pre_process = tf.Compose([
            tf.ToTensor()
        ])

    def detect(self, image: utils.ImageType, confidence=0.25, nms_thres=0.4):
        m = _bench.measurements()
        t0 = time.time()

        img = utils.ToPILImage()(image)

        t0_1 = time.time()

        img = utils.SIMDResize((self.img_size, self.img_size))(img)

        t0_2 = time.time()

        img_input = self.pre_process(img).unsqueeze(0)

        t1 = time.time()

        device = next(self.parameters()).device
        img_input = img_input.to(device)

        t2 = time.time()

        with torch.no_grad():
            detections = self(img_input)

            t3 = time.time()

            detections = non_max_suppression(detections, 80, confidence, nms_thres)

            t4 = time.time()

            timings = {
                "to_image": t0_1 - t0,
                "resize": t0_2 - t0_1,
                "pre_process": t1 - t0_2,
                "gpu_transfer": t2 - t1,
                "predict": t3 - t2,
                "nms": t4 - t3,
                "total": t4 - t0,
            }
            (m
             .add("to.image", timings["to_image"])
             .add("resize", timings["resize"])
             .add("pre.process", timings["pre_process"])
             .add("gpu.transfer", timings["gpu_transfer"])
             .add("region.proposal", timings["predict"])
             .add("nms", timings["nms"])
             .add("iteration", timings["total"])).done()
            return detections, timings

    @classmethod
    def pre_trained(cls, weights_file: str = None):
        if weights_file is None:
            weights_file = resources.weight("yolov3.weights")
        utils.downloader.download_sync(YOLOV3_WEIGHTS_URL, weights_file)
        net = cls()
        net.load_weights(weights_file)
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
                if bbox_iou(box_i, box_j) > nms_thresh:
                    box_j[4] = 0
    return out_boxes


def bbox_iou(box1, box2):
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
