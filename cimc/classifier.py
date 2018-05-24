import functools as f
import logging
import operator as op
import os
import pickle
import time
from typing import List, Optional, Dict

import imageio
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms import transforms as tf
from tqdm import tqdm

from cimc import resources, utils
from cimc.core import bbox, log
from cimc.models import YoloV3
from cimc.models.labels import COCO_LABELS
from cimc.scene import SceneDetector
from cimc.scene.classification import SceneClassifier, SceneClassification
from cimc.tracker import TrackedBoundingBox, MultiTracker
from yolo_test import make_class_labels, draw_tracked

logger = logging.getLogger(__name__)
logger.addHandler(log.TqdmLoggingHandler())
logger.setLevel(logging.DEBUG)


class Segment:
    __slots__ = ['start', 'end', 'scene', 'objects']

    def __init__(self, start: int):
        self.start = start
        self.end: Optional[int] = None
        self.scene: Optional[SceneClassification] = None
        self.objects: List[Dict[int, List[TrackedBoundingBox]]] = []

    def __len__(self):
        if self.end is None:
            raise ValueError("Segment has no 'end' frame")
        return self.end - self.start

    def add_objects(self, objects: Dict[int, List[TrackedBoundingBox]]):
        self.objects.append(objects)


def classify_video(video_uri: str):
    logger.info(f"Starting classification for video '{video_uri}'")
    segments: List[Segment] = []
    video: imageio.core.Format.Reader
    with imageio.get_reader(video_uri) as video:
        meta = video.get_meta_data()
        size = meta['size']
        fps = meta['fps']
        length = len(video)
        duration = utils.duration_str(length / fps)

        logger.info(f"'{os.path.basename(video_uri)}' : "
                    f"{size[0]}x{size[1]} {duration}({fps}fps {length} frames)")

        detections_uri = os.path.splitext(video_uri)[0] + '.dets'
        detections: List[List[bbox.BoundingBox]] = []
        gen_detections = True
        if os.path.isfile(detections_uri):
            logger.debug(f"Found detections file {detections_uri}")
            try:
                with open(detections_uri, 'rb') as dets_fd:
                    detections = pickle.load(dets_fd)
                gen_detections = False
            except pickle.UnpicklingError as e:
                logger.warning(f"The detections file {detections_uri} "
                               f"is invalid. YOLOv3 will be used")

        scene_detector = SceneDetector(downscale=4,
                                       min_length=int(fps * 2))

        scene_classifier = SceneClassifier(step=int(1 * fps))

        yolov3_net: Optional[YoloV3] = None
        if gen_detections:
            yolov3_net = YoloV3.pre_trained().to(utils.best_device)
            pp = tf.Compose([bbox.ReverseScale(size[0], size[1]),
                             bbox.FromYoloOutput(COCO_LABELS)])
        tracker = MultiTracker(max_age=int(fps),
                               min_hits=int(fps / 2),
                               iou_thres=0.35)

        t_start = time.time()
        with tqdm(total=length,
                  desc=f"Classifying '{video_uri}'",
                  dynamic_ncols=True,
                  unit='frame') as bar:
            segment: Optional[Segment] = None
            for i, frame in enumerate(video):
                bar.update()
                if scene_detector.update(frame):
                    if segment is not None:
                        segment.end = i
                        segment.scene = scene_classifier.classification()
                        segments.append(segment)
                        scene_classifier.reset()
                        tracker.reset()
                    segment = Segment(start=i)
                scene_classifier.update(frame)
                if gen_detections:
                    bboxes = [pp(box) for box in yolov3_net.detect(frame)[0]]
                    detections.append(bboxes)
                else:
                    bboxes = detections[i]
                objects = tracker.update(bboxes)
                segment.add_objects(objects)
            segment.end = length
            segment.scene = scene_classifier.classification()
            segments.append(segment)
        t_end = time.time()

        if gen_detections:
            with open(detections_uri, 'wb') as dets_fd:
                pickle.dump(detections, dets_fd)
            logger.debug(f"Saved YOLOv3 detections as '{detections_uri}'")

        time_taken = utils.duration_str(t_end - t_start)
        logger.info(f"Finished classifying '{video_uri}' in {time_taken} (found {len(segments)} scenes)")
    return segments


def draw_video_classification(video_uri: str, segments):
    import os.path
    video: imageio.core.Format.Reader
    with imageio.get_reader(video_uri) as video:
        fps = video.get_meta_data()['fps']
        length = len(video)
        out_uri, ext = os.path.splitext(video_uri)
        out_uri = f"{out_uri}-classification{ext}"
        out: imageio.core.Format.Writer
        with imageio.get_writer(out_uri, fps=fps, quality=5) as out:
            with tqdm(total=length,
                      desc=f"Drawing classification for '{video_uri}'",
                      unit='frame') as bar:
                class_colors = make_class_labels(COCO_LABELS)
                seg_iter = iter(segments)
                curr_segment: Segment = next(seg_iter)
                curr_overlay: Image.Image = None
                for i, frame in enumerate(video):
                    bar.update()
                    if i >= curr_segment.end:
                        curr_segment = next(seg_iter)
                        curr_overlay = None
                    if curr_overlay is None:
                        curr_overlay = scene_class_overlay(frame, curr_segment.scene)
                    rel_frame_idx = i - curr_segment.start
                    out_img = Image.fromarray(frame)
                    objects = curr_segment.objects[rel_frame_idx]
                    objects = f.reduce(op.concat, objects.values(), [])
                    out_img = draw_tracked(out_img,
                                           objects,
                                           class_colors)
                    out_img.paste(curr_overlay, mask=curr_overlay)
                    out_img = np.array(out_img)
                    if curr_segment.start == i:
                        out_img[-40:, -40] = [255, 0, 0]
                    out.append_data(out_img)


font = ImageFont.truetype(resources.font('DejaVuSans-Bold.ttf'), 10)


def scene_class_overlay(frame: np.ndarray, classification: SceneClassification) -> Image:
    h, w = frame.shape[:2]
    text = f"Type: {classification.type.name}\n" \
           f"Categories:\n"
    for cat in classification.categories:
        text += f"  - {cat.label}({cat.confidence*100:.2f}%)\n"
    text += f"Attributes:\n"
    for attr in classification.attributes:
        text += f"  - {attr.label}\n"
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    lines = text.splitlines()
    max_line_width = max((len(line) for line in lines))
    txt_w, txt_h = max_line_width * 7, len(lines) * 15
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, 0, txt_w, txt_h], fill=(0, 0, 0, 100))
    draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))
    return overlay


def main(video_uri=None):
    if video_uri is None:
        video_uri = resources.video('goldeneye.mp4')
    clsf_uri = video_uri + '.clsf'
    if os.path.isfile(clsf_uri):
        with open(clsf_uri, 'rb') as fd:
            segments = pickle.load(fd)
            logger.info(f"Loaded existing classification from '{clsf_uri}'")
    else:
        segments = classify_video(video_uri)
        with open(clsf_uri, 'wb') as fd:
            pickle.dump(segments, fd)
            logger.info(f"Saved classification to '{clsf_uri}'")
    draw_video_classification(video_uri, segments)


if __name__ == '__main__':
    # main(resources.video('Venice-1.mp4'))
    main(resources.video('goldeneye.mp4'))
    main(resources.video('bvs.mp4'))
    main(resources.video('justice-league.mp4'))
