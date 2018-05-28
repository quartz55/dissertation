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
from cimc.models import YoloV3
from cimc.models.yolov3.labels import COCO_LABELS
from cimc.scene import SceneDetector
from cimc.scene.classification import SceneClassifier, SceneClassification
from cimc.tracker import TrackedBoundingBox, MultiTracker
from cimc.utils import bbox, log
from yolo_test import make_class_labels, draw_tracked

logger = logging.getLogger(__name__)
logger.addHandler(log.TqdmLoggingHandler())
logger.setLevel(logging.DEBUG)


class Segment:
    __slots__ = ['id', 'start', 'end', 'scene', 'objects']

    def __init__(self, id: int, start: int):
        self.id = id
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


class VideoClassification:
    def __init__(self, filename: str,
                 segments: List[Segment],
                 length: int = 0,
                 fps: float = 0,
                 name: str = None):
        filename = os.path.basename(filename)
        if name is None:
            name = os.path.splitext(filename)[0]
        self.filename = filename
        self.name = name
        self.fps = fps
        self.length = length
        self.segments = segments

    def __iter__(self):
        return iter(self.segments)


def classify_video(video_uri: str) -> VideoClassification:
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

        detections_uri = f"{video_uri}.dets"
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
                                       min_length=int(fps))

        scene_classifier = SceneClassifier(step=int(fps / 2))

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
                    segment = Segment(id=len(segments), start=i)
                    scene_classifier.reset()
                    tracker.reset()
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
        clsf = VideoClassification(video_uri, segments, length, fps)
        return clsf


def annotate_video(video_uri: str, clsf: VideoClassification):
    import os.path
    video: imageio.core.Format.Reader
    with imageio.get_reader(video_uri) as video:
        fps = video.get_meta_data()['fps']
        length = len(video)
        out_uri, ext = os.path.splitext(video_uri)
        out_uri = f"{out_uri}.annotated.mp4"
        out: imageio.core.Format.Writer
        with imageio.get_writer(out_uri, fps=fps, quality=5, macro_block_size=2) as out:
            with tqdm(total=length,
                      desc=f"Drawing classification for '{video_uri}'",
                      unit='frame') as bar:
                class_colors = make_class_labels(COCO_LABELS)
                seg_iter = iter(clsf)
                curr_segment: Segment = next(seg_iter)
                curr_overlay: Image.Image = None
                for i, frame in enumerate(video):
                    bar.update()
                    if i >= curr_segment.end:
                        curr_segment = next(seg_iter)
                        curr_overlay = None
                    if curr_overlay is None:
                        curr_overlay = scene_class_overlay(frame, curr_segment)
                    rel_frame_idx = i - curr_segment.start
                    out_img = Image.fromarray(frame)
                    objects = curr_segment.objects[rel_frame_idx]
                    objects = f.reduce(op.concat, objects.values(), [])
                    out_img = draw_tracked(out_img,
                                           objects,
                                           class_colors)
                    out_img.paste(curr_overlay, mask=curr_overlay)
                    out_img = np.array(out_img)
                    start, end = curr_segment.start, curr_segment.end
                    scene_change_bleep_duration = min(int(fps / 2), end - start)
                    if i <= curr_segment.start < i + scene_change_bleep_duration:
                        out_img[-40:, -40:] = [255, 0, 0]
                    out.append_data(out_img)


font = ImageFont.truetype(resources.font('DejaVuSans-Bold.ttf'), 10)


def scene_class_overlay(frame: np.ndarray,
                        segment: Segment) -> Image:
    h, w = frame.shape[:2]
    start, end = segment.start, segment.end
    scene = segment.scene
    length, n_measures = scene.length, scene.num_measures

    text = f"Segment: {segment.id}({start}-{end}) {end-start} frames({length} read, {n_measures} measured)\n" \
           f"Type: {scene.type.name}\n" \
           f"Categories:\n"
    for cat in scene.categories:
        text += f"  - {cat.label}({cat.confidence*100:.2f}%)\n"
    text += f"Attributes:\n"
    for attr in scene.attributes:
        text += f"  - {attr.label}\n"

    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    lines = text.splitlines()
    max_line_width = max((len(line) for line in lines))
    txt_w, txt_h = max_line_width * 7, len(lines) * 15
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
    annotate_video(video_uri, segments)


if __name__ == '__main__':
    # try:
    #     with open(resources.video('justice-league.mp4.clsf'), 'rb') as fd:
    #         clsf = pickle.load(fd)
    #         pass
    # except:
    #     pass
    main(resources.video('TUD-Campus.var.rotate-scale.mp4'))
    # main(resources.video('TUD-Campus.mp4'))
    # main(resources.video('TUD-Crossing.mp4'))
    # main(resources.video('ADL-Rundle-8.mp4'))
    # main(resources.video('Venice-1.mp4'))
    # main(resources.video('justice-league.mp4'))
    # main(resources.video('deadpool2.mp4'))
    # main(resources.video('ant-man-and-wasp.mp4'))
    # main(resources.video('bvs.mp4'))
    # main(resources.video('goldeneye.mp4'))
