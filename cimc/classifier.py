from typing import Tuple, List

import imageio
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from cimc import resources, utils
from cimc.core import BoundingBox, bbox
from cimc.scene import SceneDetector
from cimc.tracker import TrackedBoundingBox, MultiTracker
from cimc.models import YoloV3
from cimc.models.labels import COCO_LABELS
from cimc.scene.classification import SceneClassification

import attr


def run(self):
    net = YoloV3.pre_trained(resources.weight('yolov3.weights'))
    net.to(utils.best_device)
    with imageio.get_reader(self.video_uri) as reader:
        size = reader.get_meta_data()['size']
        pp = transforms.Compose([bbox.ReverseScale(*size),
                                 bbox.FromYoloOutput(COCO_LABELS)])
        with tqdm(reader, f"Running object detection on '{self.video_uri}'",
                  unit='frame',
                  dynamic_ncols=True) as bar:
            for frame in bar:
                boxes = net.detect_image(Image.fromarray(frame))[0]
                bboxes = [pp(box) for box in boxes]
                self.detections.append(bboxes)

@attr.s(slots=True)
class Segment:
    range: Tuple[int, int] = attr.ib()
    scene: SceneClassification = attr.ib()
    objects: List[TrackedBoundingBox] = attr.ib(factory=list)

def classify_video(video_uri: str):
    segments = []
    video: imageio.core.Format.Reader
    with imageio.get_reader(video_uri) as video:
        fps = video.get_meta_data()['fps']
        length = len(video)
        with tqdm(total=length,
                  desc=f"Classifying '{video_uri}'",
                  dynamic_ncols=True,
                  unit='frame') as bar:
            scene_detector = SceneDetector(downscale=4,
                                           min_length=int(fps / 2))
            tracker = MultiTracker(max_age=int(fps),
                                   min_hits=int(fps / 2),
                                   iou_thres=0.35)