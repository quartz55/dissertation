import functools as f
import operator as op
import os.path
import re
from typing import Dict, Tuple, List, Union

import imageio
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont
from imageio.core import Format
from tqdm import tqdm

from cimc import resources
from cimc.models.yolov3.labels import COCO_LABELS
from cimc.tracker import TrackedBoundingBox
from cimc.utils.bbox import Point, BoundingBox
from .classification import VideoClassification, Segment

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

try:
    font = ImageFont.truetype(resources.font("DejaVuSansMono.ttf"), 10)
    font_bold = ImageFont.truetype(resources.font("DejaVuSansMono-Bold.ttf"), 10)
    overlay_font = ImageFont.truetype(resources.font("DejaVuSans-Bold.ttf"), 10)
except (FileNotFoundError, OSError):
    font = ImageFont.load_default()
    font_bold = ImageFont.load_default()
    overlay_font = ImageFont.load_default()


def color_luminance(color: Color) -> float:
    aux = np.array(color, dtype=np.float32)
    aux /= 255.0
    mask = aux <= 0.03928
    aux[mask] /= 12.92
    aux[~mask] = ((aux[~mask] + 0.055) / 1.055) ** 2.4
    r, g, b = aux
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


class ClassLabel:
    __slots__ = ["name", "color", "text_color"]

    def __init__(self, name: str = None, color: Color = ImageColor.getrgb("red")):
        self.name = name
        self.color = color
        lumi = color_luminance(color)
        self.text_color = ImageColor.getrgb("black" if lumi > 0.179 else "white")

    @classmethod
    def from_hue(cls, name: str, hue: int):
        return cls(cls.cap_name(name), ImageColor.getrgb(f"hsl({hue}, 100%, 50%)"))

    @staticmethod
    def cap_name(name: str) -> str:
        return re.sub("(^|\s)(\S)", lambda m: m.group(1) + m.group(2).upper(), name)


def make_class_labels(labels: List[str]) -> Dict[int, ClassLabel]:
    step = int(360 / len(labels))
    return {i: ClassLabel.from_hue(name, i * step) for i, name in enumerate(labels)}


def draw_detections(
        image: Image.Image,
        bboxes: List[BoundingBox],
        class_colors: Dict[int, ClassLabel] = None,
        thickness: float = 2,
        text_pad: float = 2
) -> Image.Image:
    if len(bboxes) == 0:
        return image

    if class_colors is None:
        class_colors = {}
    w, h = image.width, image.height
    draw = ImageDraw.Draw(image)

    for box in bboxes:
        label = class_colors.get(box.class_id, ClassLabel(color=(32, 32, 32)))
        thick = thickness
        top_left = Point.max(box.top_left, Point(0, 0))
        bot_right = Point.min(box.bot_right, Point(w - 1, h - 1))
        top_right = Point(bot_right.x, top_left.y)
        bot_left = Point(top_left.x, bot_right.y)

        # Bounding box outline
        draw.rectangle([tuple(top_left), tuple(top_right + Point(y=thick))], fill=label.color)
        draw.rectangle([tuple(top_right - Point(x=thick)), tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(bot_left - Point(y=thick)), tuple(bot_right)], fill=label.color)
        draw.rectangle([tuple(top_left), tuple(bot_left + Point(x=thick))], fill=label.color)

        if label.name is not None:
            if isinstance(box, TrackedBoundingBox):
                box: TrackedBoundingBox
                text = f"{label.name}[{box.tracking_id}]"
            else:
                text = f"{label.name}({box.confidence*100:.0f}%)"

            pad = text_pad
            text_w, text_h = font.getsize(text)
            top_left = Point.max(top_left - Point(y=text_h + pad * 2 - thick), Point(0, 0))
            bot_right = top_left + Point(x=text_w + 2 * pad, y=text_h + 2 * pad)
            draw.rectangle([tuple(top_left), tuple(bot_right)], fill=label.color)
            draw.text(tuple(top_left + Point(pad, pad)), text, fill=label.text_color, font=font_bold)
    return image


def annotate_video(video_uri: str, clsf: VideoClassification):
    with imageio.get_reader(video_uri) as video:  # type: Format.Reader

        fps = video.get_meta_data()["fps"]
        length = len(video)
        out_uri, ext = os.path.splitext(video_uri)
        out_uri = f"{out_uri}.annotated.mp4"

        with imageio.get_writer(out_uri, fps=fps, quality=5, macro_block_size=2) as out:  # type: Format.Writer
            with tqdm(total=length, desc=f"Drawing classification for '{video_uri}'", unit="frame") as bar:

                class_colors = make_class_labels(COCO_LABELS)
                seg_iter = iter(clsf)
                curr_segment: Segment = next(seg_iter)
                curr_overlay: Image.Image = None

                for i, frame in enumerate(video):
                    if i >= curr_segment.end:
                        curr_segment = next(seg_iter)
                        curr_overlay = None

                    if curr_overlay is None:
                        curr_overlay = scene_class_overlay(frame, curr_segment)

                    # Tracked objects
                    rel_frame_idx = i - curr_segment.start
                    objects = curr_segment.objects[rel_frame_idx]
                    objects = f.reduce(op.concat, objects.values(), [])

                    # Scene recognition overlay
                    out_img = Image.fromarray(frame)
                    out_img = draw_detections(out_img, objects, class_colors)
                    out_img.paste(curr_overlay, mask=curr_overlay)
                    out_img = np.array(out_img)

                    # Scene cut corner bleep
                    start, end = curr_segment.start, curr_segment.end
                    scene_change_bleep_duration = min(int(fps / 2), end - start)
                    if i <= curr_segment.start < i + scene_change_bleep_duration:
                        out_img[-40:, -40:] = [255, 0, 0]

                    out.append_data(out_img)
                    bar.update()


def scene_class_overlay(frame: np.ndarray, segment: Segment) -> Image:
    h, w = frame.shape[:2]
    start, end = segment.start, segment.end
    scene = segment.scene
    length, n_measures = scene.length, scene.num_measures

    text = f"Segment: {segment.id}({start}-{end}) {end-start} frames({length} read, {n_measures} measured)\n" f"Type: {scene.type.name}\n" f"Categories:\n"
    for _, label, conf in scene.categories[:5]:
        text += f"  - {label}({conf*100:.2f}%)\n"
    text += f"Attributes:\n"
    for _, label, freq in scene.attributes[:10]:
        text += f"  - {label}({freq*100:.2f}%)\n"

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    lines = text.splitlines()
    max_line_width = max((len(line) for line in lines))
    txt_w, txt_h = max_line_width * 7, len(lines) * 15
    draw.rectangle([0, 0, txt_w, txt_h], fill=(0, 0, 0, 100))
    draw.text((10, 10), text, font=overlay_font, fill=(255, 255, 255, 255))
    return overlay
