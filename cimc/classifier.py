from typing import Tuple, List

import attr
import imageio
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms import transforms as tf
from tqdm import tqdm

from cimc import resources, utils
from cimc.core import bbox
from cimc.models import YoloV3
from cimc.models.labels import COCO_LABELS
from cimc.scene import SceneDetector
from cimc.scene.classification import SceneClassifier, SceneClassification
from cimc.tracker import TrackedBoundingBox, MultiTracker
from yolo_test import make_class_labels, draw_tracked


@attr.s(slots=True)
class Segment:
    range: Tuple[int, int] = attr.ib()
    scene: SceneClassification = attr.ib()
    objects: List[TrackedBoundingBox] = attr.ib(factory=list)


def classify_video(video_uri: str):
    segments = []
    video: imageio.core.Format.Reader
    with imageio.get_reader(video_uri) as video:
        meta = video.get_meta_data()
        size = meta['size']
        fps = meta['fps']
        length = len(video)
        with tqdm(total=length,
                  desc=f"Classifying '{video_uri}'",
                  dynamic_ncols=True,
                  unit='frame') as bar:
            scene_detector = SceneDetector(downscale=4,
                                           min_length=int(fps / 2))
            scene_classifier = SceneClassifier(step=int(1 * fps))
            yolov3_net = YoloV3.pre_trained().to(utils.best_device)
            pp = tf.Compose([bbox.ReverseScale(size[0], size[1]),
                             bbox.FromYoloOutput(COCO_LABELS)])
            tracker = MultiTracker(max_age=int(fps),
                                   min_hits=int(fps / 2),
                                   iou_thres=0.35)
            scene = None
            for i, frame in enumerate(video):
                bar.update()
                if scene_detector.update(frame):
                    if scene is not None:
                        scene['end'] = i
                        scene['classification'] = scene_classifier.classification()
                        scene_classifier.reset()
                        tracker.reset()
                        segments.append(scene)
                    scene = {
                        'start': i,
                        'end': None,
                        'classification': None,
                        'objects': []
                    }
                scene_classifier.update(frame)
                bboxes = [pp(box) for box in yolov3_net.detect(frame)[0]]
                objects = tracker.update(bboxes)
                scene['objects'].append(objects)
            scene['end'] = length
            scene['classification'] = scene_classifier.classification()
            segments.append(scene)
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
                seg_iter = iter(segments)
                curr_segment = next(seg_iter)
                curr_overlay = None
                class_colors = make_class_labels(COCO_LABELS)
                for i, frame in enumerate(video):
                    bar.update()
                    if i >= curr_segment['end']:
                        curr_segment = next(seg_iter)
                        curr_overlay = None
                    if curr_overlay is None:
                        curr_overlay = scene_class_overlay(frame, curr_segment['classification'])
                    rel_frame_idx = i - curr_segment['start']
                    out_img = Image.fromarray(frame)
                    out_img = draw_tracked(out_img,
                                           curr_segment['objects'][rel_frame_idx],
                                           class_colors)
                    out_img.paste(curr_overlay, mask=curr_overlay)
                    out.append_data(np.array(out_img))


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


if __name__ == '__main__':
    video_uri = resources.video('goldeneye.mp4')
    segments = classify_video(video_uri)
    draw_video_classification(video_uri, segments)
