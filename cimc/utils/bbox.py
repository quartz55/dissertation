from typing import Optional, Tuple, List

import numpy as np
from numba import njit, double

from cimc.utils.vec import Vec2


class Point(Vec2):
    dtype = float


@njit(double(double[:], double[:]))
def iou(bb_a, bb_b):
    """
    Calculates Intersection over Union (IoU)

    :param bb_a: Bounding box in the format [x1, y1, x2, y2]
    :type bb_a: List[float]
    :param bb_b: Bounding box in the format [x1, y1, x2, y2]
    :type bb_b: List[float]
    :return: IoU of boxes (percentage)
    :rtype: float
    """
    a_x1, a_y1, a_x2, a_y2 = bb_a[:4]
    b_x1, b_y1, b_x2, b_y2 = bb_b[:4]
    i_x1 = max(a_x1, b_x1)
    i_y1 = max(a_y1, b_y1)
    i_x2 = min(a_x2, b_x2)
    i_y2 = min(a_y2, b_y2)

    if i_x2 < i_x1 or i_y2 < i_y1:
        return 0.0

    i_area = (i_x2 - i_x1) * (i_y2 - i_y1)
    bb_a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    bb_b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    out = i_area / float(bb_a_area + bb_b_area - i_area)
    assert 0.0 <= out <= 1.0
    return out


class BoundingBox:
    __slots__ = ['_data', 'class_name']

    def __init__(self, box: Tuple[Point, Point],
                 class_id: int = -1, name: str = None,
                 confidence: float = -1):
        self._data: np.ndarray = np.array([*box[0], *box[1],
                                           confidence, class_id],
                                          dtype=np.double)
        self.class_name: Optional[str] = name

    @property
    def top_left(self) -> Point:
        return Point(self._data[:2])

    @top_left.setter
    def top_left(self, point: Point):
        self._data[:2] = [point.x, point.y]

    @property
    def bot_right(self) -> Point:
        return Point(self._data[2:4])

    @bot_right.setter
    def bot_right(self, point: Point):
        self._data[2:4] = [point.x, point.y]

    @property
    def confidence(self) -> float:
        return self._data[4]

    @confidence.setter
    def confidence(self, value):
        self._data[4] = value

    @property
    def class_id(self) -> int:
        return int(self._data[5])

    @class_id.setter
    def class_id(self, value):
        self._data[5] = int(value)

    @property
    def mid_point(self) -> Point:
        return self.top_left + Point(self.width, self.height) / 2

    @property
    def width(self) -> float:
        return self._data[2] - self._data[0]

    @property
    def height(self) -> float:
        return self._data[3] - self._data[1]

    @property
    def area(self):
        return self.width * self.height

    def iou(self, other) -> float:
        return iou(self._data, other._data)

    def numpy(self):
        return self._data.copy()

    @classmethod
    def from_array(cls, a, labels: List[str] = None):
        """In the format [x1, y1, x2, y2] (top left and bottom right)"""
        assert len(a) >= 4, "Array needs to have at least 4 elements"
        tl, br = Point(a[0], a[1]), Point(a[2], a[3])
        conf = a[4] if len(a) > 4 else None
        class_id = a[5] if len(a) > 5 else None
        class_name = labels[class_id] if labels and class_id else None
        return cls((tl, br), class_id, class_name, conf)

    @classmethod
    def from_yolo(cls, box: List[float], labels: List[str] = None):
        """In the format [x, y, w, h] (center with width and height)"""
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2
        class_id = int(box[5])
        class_name = labels[class_id] if labels is not None else None
        return cls((Point(x1, y1), Point(x2, y2)), class_id, class_name, box[4].item())

    def __repr__(self):
        tx, ty = self.top_left
        bx, by = self.bot_right
        i = self.class_id
        name = self.class_name
        conf = self.confidence
        return f"[{i}:{name}({conf * 100:.2f}%)] ({tx:.2f},{ty:.2f})({bx:.2f},{by:.2f})"

    def __str__(self):
        return self.__repr__()


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


class FromYoloOutput:
    def __init__(self, labels: List[str] = None):
        self.labels = labels

    def __call__(self, box: List[float]) -> BoundingBox:
        return BoundingBox.from_yolo(box, self.labels)
