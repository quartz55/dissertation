from typing import Optional, Tuple, List
from brambox.boxes.detections import Detection
from cimc.core.vec import Vec2


class Point(Vec2):
    dtype = int


class BoundingBox:
    __slots__ = ['top_left', 'bot_right', 'class_id', 'class_name', 'confidence']

    def __init__(self, box: Tuple[Point, Point],
                 class_id: int = -1, name: str = None,
                 confidence: float = 0):
        self.top_left: Point = box[0]
        self.bot_right: Point = box[1]
        self.class_id: int = class_id
        self.class_name: Optional[str] = name
        self.confidence: float = confidence

    @classmethod
    def from_yolo(cls, box: List[float], labels: List[str] = None):
        x1 = int(box[0] - box[2] / 2)
        y1 = int(box[1] - box[3] / 2)
        x2 = int(box[0] + box[2] / 2)
        y2 = int(box[1] + box[3] / 2)
        class_id = int(box[5])
        class_name = labels[class_id] if labels is not None else None
        return cls((Point(x1, y1), Point(x2, y2)), class_id, class_name, box[4])

    @classmethod
    def from_brambox(cls, bbox: Detection, labels: List[str] = None):
        top_left = Point(bbox.x_top_left, bbox.y_top_left)
        bot_right = top_left + Point(bbox.width, bbox.height)
        class_id = int(bbox.class_label)
        class_name = labels[class_id] if labels is not None else None
        return cls((top_left, bot_right), class_id, class_name, bbox.confidence)

    @property
    def mid_point(self) -> Point:
        return self.top_left + Point(self.width, self.height) / 2

    @property
    def width(self) -> int:
        return self.bot_right.x - self.top_left.x

    @property
    def height(self) -> int:
        return self.bot_right.y - self.top_left.y


class FromYoloOutput:
    def __init__(self, labels: List[str] = None):
        self.labels = labels

    def __call__(self, box: List[float]) -> BoundingBox:
        return BoundingBox.from_yolo(box, self.labels)


class ReverseScale:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, box: List[float]) -> List[float]:
        box[0] *= self.width
        box[1] *= self.height
        box[2] *= self.width
        box[3] *= self.height
        return box


class FromBramBox:
    def __init__(self, labels: List[str]):
        self.labels = labels

    def __call__(self, brambox: Detection) -> BoundingBox:
        return BoundingBox.from_brambox(brambox, self.labels)
