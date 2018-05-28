import functools as f
import operator as op
import os.path
import pickle
import re
from typing import List, Union, Tuple, Dict

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from torchvision import transforms
from tqdm import tqdm

import cimc.utils.bbox as bbox
import cimc.resources as resources
from cimc import utils
from cimc.utils.bbox import BoundingBox, Point
from cimc.models.yolov3.labels import COCO_LABELS
from cimc.models.yolov3 import YoloV3
from cimc.tracker import TrackedBoundingBox, MultiTracker

try:
    font = ImageFont.truetype(resources.font('DejaVuSansMono.ttf'), 10)
    font_bold = ImageFont.truetype(resources.font('DejaVuSansMono-Bold.ttf'), 10)
except (FileNotFoundError, OSError):
    font = ImageFont.load_default()
    font_bold = ImageFont.load_default()

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
    draw = ImageDraw.Draw(image)
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
            pad = 2
            text = f"{label.name}{box.tracking_id}"
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
    return image


class VideoDetections:
    def __init__(self, video_uri: str) -> None:
        self.video_uri = video_uri
        self.detections: List[List[BoundingBox]] = []

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
                    boxes = net.detect(Image.fromarray(frame))[0]
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


def test_tracking(dets_file: str):
    import time
    detections: VideoDetections = VideoDetections.load(dets_file)
    vid_base_uri = os.path.splitext(dets_file)[0]
    in_uri = f"{vid_base_uri}.mp4"
    out_uri = f"{vid_base_uri}-tracked.mp4"
    with imageio.get_reader(in_uri) as video:
        fps = video.get_meta_data()['fps']
        with imageio.get_writer(out_uri, fps=fps, quality=6) as writer:
            with tqdm(video,
                      f"Tracking detections of '{detections.video_uri}'",
                      unit='frame',
                      dynamic_ncols=True) as video:
                class_colors = make_class_labels(COCO_LABELS)
                tracker = MultiTracker(max_age=int(fps),
                                       min_hits=int(fps / 2),
                                       iou_thres=0.35)
                timings = np.empty(len(video))
                for frame, bboxes in zip(enumerate(video), detections.detections):
                    idx, frame = frame
                    t = time.time()
                    tracked_objects = tracker.update(bboxes)
                    tracked_objects = f.reduce(op.concat, tracked_objects.values())
                    timings[idx] = time.time() - t
                    result = draw_tracked(Image.fromarray(frame), tracked_objects, class_colors)
                    writer.append_data(np.array(result))
                mean = np.mean(timings)
                print(f"Average tracking time: {mean}s | {mean*1000}ms | {1/mean}fps")


if __name__ == '__main__':
    pass
    # gen_detections(resources.video('TUD-Campus.mp4'))
    # gen_detections(resources.video('TUD-Crossing.mp4'))
    # gen_detections(resources.video('Venice-1.mp4'))
    # gen_detections(resources.video('goldeneye.mp4'))
    test_tracking(resources.video('goldeneye.dets'))
    test_tracking(resources.video('TUD-Campus.dets'))
    test_tracking(resources.video('TUD-Crossing.dets'))
    test_tracking(resources.video('Venice-1.dets'))
    # pr = cProfile.Profile()
    # pr.enable()
    # test_tracking()
    # pr.disable()
    # pr.dump_stats('tracking.pstats')
