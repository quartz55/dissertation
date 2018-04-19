from PIL import Image, ImageDraw, ImageFont, ImageColor

from typing import List, Union, Tuple, Dict
import imageio
import numpy as np
from tqdm import tqdm

from cimc.models.yolov3 import COCO_LABELS
import cimc.core.bbox as bbox
from cimc.core.bbox import BoundingBox, Point
import re

font = ImageFont.truetype('resources/DejaVuSansMono.ttf', 16)
font_bold = ImageFont.truetype('resources/DejaVuSansMono-Bold.ttf', 16)

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


class ClassLabel:
    __slots__ = ['name', 'color']

    def __init__(self, name: str, color: Color = ImageColor.getrgb('red')):
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
        label = class_colors.get(box.class_id, ClassLabel(None, (32, 32, 32)))
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


def test_yolo2(duration: int):
    from cimc.models.yolov2 import YoloV2

    def detect_video(net: YoloV2, video_uri: str, out_uri: str, duration: int = None):
        post_process = bbox.FromBramBox(COCO_LABELS)
        from imageio.plugins.ffmpeg import FfmpegFormat
        with imageio.get_reader(video_uri) as reader:  # type: FfmpegFormat.Reader
            fps = reader.get_meta_data()['fps']
            with imageio.get_writer(out_uri, fps=fps, quality=6) as writer:  # type: FfmpegFormat.Writer
                bar = tqdm(reader, f"Object detection using '{net.__class__.__name__}' on '{video_uri}'", unit='frame')
                frames = enumerate(bar)
                class_colors = make_class_labels(COCO_LABELS)
                frame_stop = fps * duration if duration is not None else None
                for index, frame in frames:
                    if frame_stop is not None and index >= frame_stop:
                        break
                    boxes, image, timings = net.detect_image(frame)
                    boxes = [post_process(box) for box in boxes[0]]
                    result = draw_detections(Image.fromarray(image), boxes, class_colors)
                    writer.append_data(np.array(result))
                    # info = {k: f"{t*1000:.2f}ms" for k, t in timings.items()}
                    # bar.write(str(info) + '\n' + f"{1 / timings['total']:.1f} fps")

    net = YoloV2.pre_trained('resources/yolov2.weights', confidence=0.25)
    net.cuda()
    detect_video(net, 'resources/goldeneye.mp4', 'goldeneye-yolo2.mp4', duration)


def test_yolo3(duration: int = None):
    from cimc.models.yolov3 import YoloV3
    from torchvision import transforms

    def detect_video(net: YoloV3, video_uri: str, out_uri: str, duration: int = None):
        from imageio.plugins.ffmpeg import FfmpegFormat
        with imageio.get_reader(video_uri) as reader:  # type: FfmpegFormat.Reader
            fps = reader.get_meta_data()['fps']
            size: Tuple[int, int] = reader.get_meta_data()['size']
            post_process = transforms.Compose([bbox.ReverseScale(*size), bbox.FromYoloOutput(COCO_LABELS)])
            with imageio.get_writer(out_uri, fps=fps, quality=6) as writer:  # type: FfmpegFormat.Writer
                bar = tqdm(reader, f"Object detection using '{net.__class__.__name__}' on '{video_uri}'", unit='frame')
                frames = enumerate(bar)
                frame_stop = fps * duration if duration is not None else None
                class_colors = make_class_labels(COCO_LABELS)
                for index, frame in frames:
                    if frame_stop is not None and index >= frame_stop:
                        break
                    boxes, image, timings = net.detect_image(Image.fromarray(frame))
                    bboxes = [post_process(box) for box in boxes]
                    result = draw_detections(image, bboxes, class_colors)
                    writer.append_data(np.array(result))
                    # result.show()
                    # input('Press ENTER to continue')
                    # info = {k: f"{t*1000:.2f}ms" for k, t in timings.items()}
                    # bar.write(str(info) + '\n' + f"{1 / timings['total']:.1f} fps")

    net = YoloV3.pre_trained('resources/yolov3.weights')
    net.cuda()
    detect_video(net, 'resources/goldeneye.mp4', 'goldeneye-yolo3.mp4', duration)
    # detect_video(net, 'resources/bvs.mp4', 'bvs-yolo3.mp4')
    # detect_video_custom(net, 'resources/goldedRneye.mp4')
    # detect_and_show(net, 'resources/people-2.jpg')


def main():
    import argparse
    import cProfile

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        default='yolo3', choices=['yolo2', 'yolo3'],
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

    if args.model == 'yolo2':
        test_yolo2(args.duration)
    elif args.model == 'yolo3':
        test_yolo3(args.duration)

    if prof is not None:
        prof.disable()
        prof.dump_stats(args.profile)


if __name__ == '__main__':
    main()
