import re
from typing import List, Union, Tuple, Dict, Any

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageColor
from tqdm import tqdm

import cimc.core.bbox as bbox
from cimc.core.bbox import BoundingBox, Point
from cimc.core.sort import Sort
from cimc.models.labels import COCO_LABELS

try:
    font = ImageFont.truetype('resources/fonts/DejaVuSansMono.ttf', 14)
    font_bold = ImageFont.truetype('resources/fonts/DejaVuSansMono-Bold.ttf', 14)
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
        thick = 5
        top_left = Point.max(box.top_left, Point(0, 0))
        bot_right = Point.min(box.bot_right, Point(image.width - 1, image.height - 1))
        top_right = Point(bot_right.x, top_left.y)
        bot_left = Point(top_left.x, bot_right.y)
        # Draw rectangle outline manually
        draw.rectangle([tuple(top_left), tuple(top_right + Point(y=thick))], fill=label.color)
        draw.rectangle([tuple(top_right - Point(x=thick)), tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(bot_left - Point(y=thick)), tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(top_left), tuple(bot_left + Point(x=thick))], fill=label.color)
        if label.name is not None:
            pad = 5
            text = f"{label.name}({box.confidence*100:.0f}%)"
            text_w, text_h = font.getsize(text)
            top_left = Point.max(top_left - Point(y=text_h + pad), Point(0, 0))
            bot_right = top_left + Point(x=text_w + 2 * pad, y=text_h + 2 * pad)
            draw.rectangle([tuple(top_left),
                            tuple(bot_right)],
                           fill=label.color)
            draw.text(tuple(top_left + Point(pad, pad)), text, fill='white', font=font_bold)
    return result


def draw_tracked(image: Image.Image, bboxes: np.ndarray,
                 class_colors: Dict[int, ClassLabel] = None) -> Image.Image:
    if len(bboxes) == 0:
        return image
    if class_colors is None:
        class_colors = {}
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for box in bboxes:
        label = class_colors.get(box[5], ClassLabel(color=(32, 32, 32)))
        thick = 2
        top_left = Point.max(Point(box[0], box[1]), Point(0, 0))
        bot_right = Point.min(Point(box[2], box[3]), Point(image.width - 1, image.height - 1))
        top_right = Point(bot_right.x, top_left.y)
        bot_left = Point(top_left.x, bot_right.y)
        # Draw rectangle outline manually
        draw.rectangle([tuple(top_left), tuple(top_right + Point(y=thick))], fill=label.color)
        draw.rectangle([tuple(top_right - Point(x=thick)), tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(bot_left - Point(y=thick)), tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(top_left), tuple(bot_left + Point(x=thick))], fill=label.color)
        if label.name is not None:
            pad = 3
            text = f"{label.name}({int(box[4])})"
            text_w, text_h = font.getsize(text)
            top_left = Point.max(top_left - Point(y=text_h + pad * 2 - thick), Point(0, 0))
            bot_right = top_left + Point(x=text_w + 2 * pad, y=text_h + 2 * pad)
            draw.rectangle([tuple(top_left),
                            tuple(bot_right)],
                           fill=label.color)
            draw.text(tuple(top_left + Point(pad, pad)), text, fill='white', font=font_bold)
    return result


class TrackedObject(bbox.BoundingBox):
    def __init__(self, ident: int,
                 box: Tuple[Point, Point],
                 class_id: int = -1, name: str = None,
                 confidence: float = 0):
        super().__init__(box, class_id, name, confidence)
        self.ident = ident

    @classmethod
    def from_bbox(cls, bbox: BoundingBox, ident: int):
        return cls(ident,
                   (bbox.top_left, bbox.bot_right),
                   bbox.class_id,
                   bbox.class_name,
                   bbox.confidence)


def test_yolo3(duration: int = None):
    from cimc.models.yolov3 import YoloV3
    from torchvision import transforms

    def detect_video(net: YoloV3, video_uri: str, out_uri: str, duration: int = None):
        from imageio.plugins.ffmpeg import FfmpegFormat
        with imageio.get_reader(video_uri) as reader:  # type: FfmpegFormat.Reader
            fps = reader.get_meta_data()['fps']
            size: Tuple[int, int] = reader.get_meta_data()['size']
            post_process = transforms.Compose([bbox.ReverseScale(*size), bbox.FromYoloOutput(COCO_LABELS)])
            frame_stop = round(fps * duration) if duration is not None else None
            with imageio.get_writer(out_uri, fps=fps, quality=6) as writer:  # type: FfmpegFormat.Writer
                with tqdm(take(reader, frame_stop or len(reader)),
                          f"Object detection using '{net.__class__.__name__}' on '{video_uri}'",
                          unit='frame',
                          dynamic_ncols=True) as bar:
                    class_colors = make_class_labels(COCO_LABELS)
                    trackers: Dict[int, Sort] = {}
                    for index, frame in enumerate(bar):
                        boxes, image, _timings = net.detect_image(Image.fromarray(frame))
                        bboxes = [post_process(box) for box in boxes]
                        bboxes_per_class = {}
                        tracked = {}
                        for box in bboxes:
                            if box.class_id in bboxes_per_class:
                                bboxes_per_class[box.class_id].append(box)
                            else:
                                bboxes_per_class[box.class_id] = [box]
                        for id, bb in bboxes_per_class.items():
                            if id not in trackers:
                                trackers[id] = Sort(max_age=int(fps), iou_thres=0.1)
                            tracked[id] = trackers[id].update(np.array([b.as_tracker() for b in bb]))[::-1]
                        # result = draw_detections(image, bboxes, class_colors)
                        tracked_list = np.concatenate([np.c_[box, np.full((len(box),), id)] for id, box in tracked.items()])
                        result = draw_tracked(image, tracked_list, class_colors)
                        writer.append_data(np.array(result))

    net = YoloV3.pre_trained('resources/weights/yolov3.weights')
    net.to(device)
    detect_video(net, 'resources/videos/ADL-Rundle-8.mp4', 'ADL-Rundle-8-yolo3.mp4', duration)


def main():
    import argparse
    import cProfile

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, nargs='?',
                        default='yolo3', choices=['yolo3'],
                        help="Model to use for detection (yolo2 or yolo3)")
    parser.add_argument('-d', '--duration', type=int,
                        help="Duration of the video to detect")
    parser.add_argument('-P', '--profile', type=str,
                        help="Outputs cProfile stats to specified file")
    args = parser.parse_args()

    prof = None
    if args.profile is not None:
        prof = cProfile.Profile()
        prof.enable()

    if args.model == 'yolo3':
        test_yolo3(args.duration)

    if prof is not None:
        prof.disable()
        prof.dump_stats(args.profile)


if __name__ == '__main__':
    main()
