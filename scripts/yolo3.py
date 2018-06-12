import os.path

import imageio
import torchvision.transforms as tf
from PIL import Image

from cimc import resources, utils
from cimc.classifier.annotator import draw_detections, make_class_labels
from cimc.models import YoloV3
from cimc.models.yolov3.labels import COCO_LABELS
from cimc.utils import bbox

CLASS_COLORS = make_class_labels(COCO_LABELS)


def yolo3_net():
    return YoloV3.pre_trained().to(utils.best_device)


def detect_and_save(img_uri: str, out_uri: str = None, net: YoloV3 = None):
    if out_uri is None:
        out_uri = ".dets".join(os.path.splitext(img_uri))

    img = imageio.imread(img_uri)
    h, w, _ = img.shape
    net = yolo3_net() if net is None else net
    pp = tf.Compose([bbox.ReverseScale(w, h), bbox.FromYoloOutput(COCO_LABELS)])
    bboxes = [pp(box) for box in net.detect(img)[0]]
    out = draw_detections(Image.fromarray(img), bboxes, CLASS_COLORS)
    # out.show()
    out.save(out_uri)


if __name__ == '__main__':
    net = yolo3_net()
    detect_and_save(resources.image("yolo3-test1.jpg"), net=net)
    detect_and_save(resources.image("yolo3-test2.jpg"), net=net)
    detect_and_save(resources.image("yolo3-test3.jpg"), net=net)
