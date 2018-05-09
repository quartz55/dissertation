import os.path
import pickle
import re
from typing import List, Union, Tuple, Dict

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageColor
from torchvision import transforms
from tqdm import tqdm
import functools as f
import operator as op

import cimc.core.bbox as bbox
from cimc.core.bbox import BoundingBox, Point
from cimc.core.tracker import TrackedBoundingBox, Tracker, MultiTracker
from cimc.models.labels import COCO_LABELS
from cimc.models.yolov3 import YoloV3


class Resources:
    root = 'resources'

    @staticmethod
    def font(name: str) -> str:
        return os.path.join(Resources.root, 'fonts', name)

    @staticmethod
    def video(name: str) -> str:
        return os.path.join(Resources.root, 'videos', name)

    @staticmethod
    def weights(name: str) -> str:
        return os.path.join(Resources.root, 'weights', name)


try:
    font = ImageFont.truetype(Resources.font('DejaVuSansMono.ttf'), 14)
    font_bold = ImageFont.truetype(Resources.font('DejaVuSansMono-Bold.ttf'), 14)
except (FileNotFoundError, OSError):
    font = ImageFont.load_default()
    font_bold = ImageFont.load_default()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


def take(gen, n=1):
    class TakeGenerator:
        def __init__(self):
            self.__gen = iter(gen)
            self.__i = 1
            self.__n = min(n, len(gen))

        def __iter__(self):
            return self

        def __next__(self):
            if self.__i > self.__n:
                raise StopIteration
            self.__i += 1
            return next(self.__gen)

        def __len__(self):
            return self.__n

    return TakeGenerator()


class ClassLabel:
    __slots__ = ['name', 'color']

    def __init__(self, name: str = None, color: Color = ImageColor.getrgb('red')):
        self.name = name
        self.color = color

    @classmethod
    def from_hue(cls, name: str, hue: int):
        return cls(cls.cap_name(name), ImageColor.getrgb(f"hsl({hue}, 100%, 50%)"))

    @staticmethod
    def cap_name(name: str) -> str:
        return re.sub("(^|\s)(\S)", lambda m: m.group(1) + m.group(2).upper(), name)


def make_class_labels(labels: List[str]) -> Dict[int, ClassLabel]:
    step = int(360 / len(labels))
    return {i: ClassLabel.from_hue(name, i * step)
            for i, name in enumerate(labels)}


def draw_detections(image: Image.Image, bboxes: List[BoundingBox],
                    class_colors: Dict[int, ClassLabel] = None) -> Image.Image:
    if len(bboxes) == 0:
        return image
    if class_colors is None:
        class_colors = {}
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for box in bboxes:
        label = class_colors.get(box.class_id, ClassLabel(color=(32, 32, 32)))
        thick = 2
        top_left = Point.max(box.top_left, Point(0, 0))
        bot_right = Point.min(box.bot_right, Point(
            image.width - 1, image.height - 1))
        top_right = Point(bot_right.x, top_left.y)
        bot_left = Point(top_left.x, bot_right.y)
        # Draw rectangle outline manually
        draw.rectangle([tuple(top_left), tuple(
            top_right + Point(y=thick))], fill=label.color)
        draw.rectangle([tuple(top_right - Point(x=thick)),
                        tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(bot_left - Point(y=thick)),
                        tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(top_left), tuple(
            bot_left + Point(x=thick))], fill=label.color)
        if label.name is not None:
            pad = 3
            text = f"{label.name}({box.confidence*100:.0f}%)"
            text_w, text_h = font.getsize(text)
            top_left = Point.max(
                top_left - Point(y=text_h + pad * 2 - thick), Point(0, 0))
            bot_right = top_left + \
                        Point(x=text_w + 2 * pad, y=text_h + 2 * pad)
            draw.rectangle([tuple(top_left),
                            tuple(bot_right)],
                           fill=label.color)
            draw.text(tuple(top_left + Point(pad, pad)),
                      text, fill='white', font=font_bold)
    return result


def draw_tracked(image: Image.Image, tracked: List[TrackedBoundingBox],
                 class_colors: Dict[int, ClassLabel] = None) -> Image.Image:
    if len(tracked) == 0:
        return image
    if class_colors is None:
        class_colors = {}
    w, h = image.width, image.height
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for box in tracked:
        label = class_colors.get(box.class_id, ClassLabel(color=(32, 32, 32)))
        thick = 2
        top_left = Point.max(box.top_left, Point(0, 0))
        bot_right = Point.min(box.bot_right, Point(w - 1, h - 1))
        top_right = Point(bot_right.x, top_left.y)
        bot_left = Point(top_left.x, bot_right.y)
        # Draw rectangle outline manually
        draw.rectangle([tuple(top_left), tuple(
            top_right + Point(y=thick))], fill=label.color)
        draw.rectangle([tuple(top_right - Point(x=thick)),
                        tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(bot_left - Point(y=thick)),
                        tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(top_left), tuple(
            bot_left + Point(x=thick))], fill=label.color)
        if label.name is not None:
            pad = 3
            text = f"{label.name} {box.tracking_id}({box.confidence:.0f})"
            text_w, text_h = font.getsize(text)
            top_left = Point.max(
                top_left - Point(y=text_h + pad * 2 - thick), Point(0, 0))
            bot_right = top_left + \
                        Point(x=text_w + 2 * pad, y=text_h + 2 * pad)
            draw.rectangle([tuple(top_left),
                            tuple(bot_right)],
                           fill=label.color)
            draw.text(tuple(top_left + Point(pad, pad)),
                      text, fill='white', font=font_bold)
    return result


class VideoDetections:
    def __init__(self, video_uri: str) -> None:
        self.video_uri = video_uri
        self.detections: List[List[BoundingBox]] = []

    def run(self):
        net = YoloV3.pre_trained(Resources.weights('yolov3.weights'))
        net.to(device)
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

    def render(self, path: str = None, quality: int = 6):
        savepath = path
        if savepath is None:
            base, ext = os.path.splitext(self.video_uri)
            savepath = f"{base}-detections{ext}"
        with imageio.get_reader(self.video_uri) as reader:
            fps = reader.get_meta_data()['fps']
            with imageio.get_writer(savepath, fps=fps, quality=quality) as writer:
                with tqdm(reader,
                          f"Rendering detections of '{self.video_uri}' to '{savepath}'",
                          unit='frame',
                          dynamic_ncols=True) as bar:
                    class_colors = make_class_labels(COCO_LABELS)
                    for frame, bboxes in zip(bar, self.detections):
                        result = draw_detections(
                            Image.fromarray(frame), bboxes, class_colors)
                        writer.append_data(np.array(result))

    def save(self, path: str = None):
        savepath = path or f"{os.path.splitext(self.video_uri)[0]}.dets"
        with open(savepath, 'wb') as out:
            pickle.dump(self, out)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as file:
            return pickle.load(file)


def gen_detections(video_file: str):
    v = VideoDetections(video_file)
    v.run()
    v.save()


def test_tracking():
    import time
    detections: VideoDetections = VideoDetections.load(Resources.video('ADL-Rundle-8.dets'))
    with imageio.get_reader(detections.video_uri) as video:
        fps = video.get_meta_data()['fps']
        with imageio.get_writer('test-tracking.mp4', fps=fps, quality=6) as writer:
            with tqdm(video,
                      f"Tracking detections of '{detections.video_uri}'",
                      unit='frame',
                      dynamic_ncols=True) as video:
                class_colors = make_class_labels(COCO_LABELS)
                tracker = MultiTracker(int(fps), iou_thres=0.2)
                timings = np.empty(len(video))
                for frame, bboxes in zip(enumerate(video), detections.detections):
                    idx, frame = frame
                    tracked_objects = tracker.update(bboxes)
                    tracked_objects = f.reduce(op.concat, tracked_objects.values())
                    t = time.time()
                    result = draw_tracked(Image.fromarray(frame), tracked_objects, class_colors)
                    timings[idx] = time.time() - t
                    writer.append_data(np.array(result))
                print(f"Average tracking time: {np.mean(timings)}")


if __name__ == '__main__':
    # gen_detections(Resources.video('ADL-Rundle-8.mp4'))
    test_tracking()
