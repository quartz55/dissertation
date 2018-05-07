from typing import List, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment

from cimc.core.bbox import BoundingBox, Point


class TrackedBoundingBox(BoundingBox):
    def __init__(self, box: Tuple[Point, Point],
                 class_id: int = -1, name: str = None,
                 confidence: float = -1,
                 tracking_id: int = -1):
        super().__init__(box, class_id, name, confidence)
        self.tracking_id = tracking_id


def convert_bbox_to_z(bbox: BoundingBox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox.width
    h = bbox.height
    x, y = bbox.mid_point
    s = w * h
    r = w / h
    return np.array([x, y, s, r]).reshape((4, 1))


class KalmanBoxTracker:
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox: BoundingBox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self._bbox: BoundingBox = bbox
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    @property
    def confidence(self):
        return self._bbox.confidence

    @property
    def class_id(self):
        return self._bbox.class_id

    @property
    def class_name(self):
        return self._bbox.class_name

    def bbox(self):
        state = self.kf.x[:4].T[0]
        w = np.sqrt(state[2] * state[3])
        h = state[2] / w
        top_left = Point(state[0] - w / 2., state[1] - h / 2.)
        bot_right = Point(state[0] + w / 2., state[1] + h / 2.)
        return TrackedBoundingBox((top_left, bot_right),
                                  self.class_id, self.class_name,
                                  self.confidence, self.id)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.bbox())
        return self.history[-1]

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))


class Tracker:
    def __init__(self, max_age=1, min_hits=3, iou_thres=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thres = iou_thres
        self.trackers: List[KalmanBoxTracker] = []
        self.curr_frame = 0

    def _match_trackers(self, bboxes: List[BoundingBox]):
        """
        Match detections to existing trackers (using IoU as metric)
        :param bboxes: List of measured bounding boxes
        :type bboxes: List[BoundingBox]
        :return: (matched, unmatched_dets, unmatched_trackers)
        """
        if len(self.trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(bboxes)), np.empty((0, 5), dtype=int)

        # Build IoU matrix between measured and predicted bboxes
        iou_matrix = np.zeros((len(bboxes), len(self.trackers)), dtype=np.float32)
        for d, det in enumerate(bboxes):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = det.iou(trk.bbox())
        matched_indices = linear_assignment(-iou_matrix)

        unmatched_detections = []
        for d, det in enumerate(bboxes):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(self.trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_thres:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, bboxes):
        self.curr_frame += 1

        # Kalman Filter predict step
        predictions = [t.predict() for t in self.trackers]

        # Match predicted bboxes to measured bboxes using IoU
        matched, unmatched_dets, unmatched_trks = self._match_trackers(bboxes)

        # Kalman Filter update step
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(bboxes[d, :][0])

        # Start tracking unmatched bboxes
        for i in unmatched_dets:
            trk = KalmanBoxTracker(bboxes[i])
            self.trackers.append(trk)

        # Cleanup zombie trackers
        ret = []
        for i, trk in reversed(list(enumerate(self.trackers))):
            d = trk.bbox()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.curr_frame <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
