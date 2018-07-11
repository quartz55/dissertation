import multiprocessing as mp
import time
from typing import List, Tuple, Dict

import attr
import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment

from cimc.utils import bench
from cimc.utils.bbox import BoundingBox, Point


class TrackedBoundingBox(BoundingBox):
    def __init__(self, box: Tuple[Point, Point],
                 class_id: int = -1, name: str = None,
                 confidence: float = -1,
                 tracking_id: int = -1):
        super().__init__(box, class_id, name, confidence)
        self.tracking_id = tracking_id

    @classmethod
    def from_bbox(cls, bbox: BoundingBox, tracking_id: int = -1):
        return cls((bbox.top_left, bbox.bot_right),
                   bbox.class_id, bbox.class_name,
                   bbox.confidence, tracking_id)

    def __repr__(self):
        return f"({self.tracking_id}){super().__repr__()}"


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


class InvalidPrediction(Exception):
    pass


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
        if np.isnan(self.kf.x).any():
            raise InvalidPrediction

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


@attr.s(slots=True)
class Matches:
    matches: np.ndarray = attr.ib(factory=lambda: np.empty((0, 2), dtype=int))
    new_matches: np.ndarray = attr.ib(factory=lambda: np.empty(0, dtype=int))
    no_matches: np.ndarray = attr.ib(factory=lambda: np.empty(0, dtype=int))


class Tracker:
    def __init__(self, max_age=1, min_hits=3, iou_thres=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thres = iou_thres
        self.trackers: List[KalmanBoxTracker] = []
        self._bench = bench.Bench("object.tracker")

    def _match_trackers(self, bboxes: List[BoundingBox]):
        """
        Match detections to existing trackers (using IoU as metric)
        :param bboxes: List of measured bounding boxes
        :type bboxes: List[BoundingBox]
        :return: (matched, unmatched_dets, unmatched_trackers)
        """
        if len(self.trackers) == 0:
            return Matches(new_matches=np.arange(len(bboxes)))

        # Build IoU matrix between measured and predicted bboxes
        iou_matrix = np.zeros((len(bboxes), len(self.trackers)),
                              dtype=np.float32)
        for d, bbox in enumerate(bboxes):
            for t, tracker in enumerate(self.trackers):
                iou_matrix[d, t] = bbox.iou(tracker.bbox())
        matches_idx = linear_assignment(-iou_matrix)

        new_matches = [d for d, det in enumerate(bboxes) if d not in matches_idx[:, 0]]
        no_matches = [t for t, tracker in enumerate(self.trackers) if t not in matches_idx[:, 1]]

        matches = []
        for m in matches_idx:
            if iou_matrix[m[0], m[1]] < self.iou_thres:
                new_matches.append(m[0])
                no_matches.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return Matches(matches, np.array(new_matches), np.array(no_matches))

    def update(self, bboxes):
        t0 = time.time()

        # Kalman Filter predict step
        for i, tracker in reversed(list(enumerate(self.trackers))):
            try:
                tracker.predict()
            except InvalidPrediction:
                self.trackers.pop(i)

        t1 = time.time()

        # Match predicted bboxes to measured bboxes using IoU
        mr = self._match_trackers(bboxes)

        t2 = time.time()

        # Kalman Filter update step
        for t, tracker in enumerate(self.trackers):
            if t not in mr.no_matches:
                d = mr.matches[np.where(mr.matches[:, 1] == t)[0], 0]
                tracker.update(bboxes[d[0]])

        t3 = time.time()

        # Start tracking unmatched bboxes
        for i in mr.new_matches:
            self.trackers.append(KalmanBoxTracker(bboxes[i]))

        ret = []
        for i, tracker in reversed(list(enumerate(self.trackers))):
            tracker: KalmanBoxTracker
            # Cleanup zombie trackers
            if tracker.time_since_update > self.max_age:
                self.trackers.pop(i)
                continue

            # Don't include uncertain trackers
            if (tracker.time_since_update > 0
                    and tracker.hits < 10
                    and tracker.bbox().area < 500):
                continue

            # if tracker.hits >= self.min_hits or self.curr_frame <= self.min_hits:
            #     ret.append(TrackedBoundingBox.from_bbox(tracker.bbox(), tracker.id))

            if (tracker.hits >= self.min_hits
                    or tracker.hit_streak >= 3):
                ret.append(TrackedBoundingBox.from_bbox(tracker.bbox(), tracker.id))

        (self._bench.measurements()
         .add("predict.step", t1 - t0)
         .add("kf.iou.match", t2 - t1)
         .add("update.step", t3 - t2)
         .add("post.process", time.time() - t3)).done()

        return ret

    def reset(self):
        self.trackers = []


class MultiTracker:
    def __init__(self, max_age=1, min_hits=3, iou_thres=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thres = iou_thres
        self.trackers: Dict[int, Tracker] = {}
        self._bench = bench.Bench("multi.tracker")

    def update(self, bboxes) -> Dict[int, List[TrackedBoundingBox]]:
        t0 = time.time()

        per_class = {cls: [] for cls in self.trackers}
        for bbox in bboxes:
            if bbox.class_id in per_class:
                per_class[bbox.class_id].append(bbox)
            else:
                per_class[bbox.class_id] = [bbox]

        t1 = time.time()

        tracked = {}
        for cls, bbs in per_class.items():
            if cls not in self.trackers:
                tracker = Tracker(self.max_age, self.min_hits, self.iou_thres)
                self.trackers[cls] = tracker
            tracked[cls] = self.trackers[cls].update(bbs)

        (self._bench.measurements()
         .add("pre.process", t1 - t0)
         .add("trackers.update", time.time() - t1)).done()

        return tracked

    def reset(self):
        self.trackers = {}

    def cleanup(self):
        pass


def tracker_process(tracker: Tracker, tasks: mp.Queue, results: mp.Queue):
    while True:
        [msg, bboxes] = tasks.get()
        if msg == "exit":
            break
        elif msg == "reset":
            tracker.reset()
        elif msg == "update":
            tracked = tracker.update(bboxes)
            results.put(tracked)


class ParallelMultiTracker:
    def __init__(self, max_age=1, min_hits=3, iou_thres=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thres = iou_thres
        self.trackers: Dict[int, Tuple[mp.Process, mp.Queue, mp.Queue]] = {}
        self._bench = bench.Bench("par.multi.tracker")

    def update(self, bboxes) -> Dict[int, List[TrackedBoundingBox]]:
        if len(bboxes) == 0:
            return {}

        t0 = time.time()

        per_class = {cls: [] for cls in self.trackers.keys()}
        for bbox in bboxes:
            if bbox.class_id in per_class:
                per_class[bbox.class_id].append(bbox)
            else:
                per_class[bbox.class_id] = [bbox]

        t1 = time.time()

        tracked = {}
        for cls, bbs in per_class.items():
            if cls not in self.trackers:
                tracker = Tracker(self.max_age, self.min_hits, self.iou_thres)
                tasks, results = mp.Queue(), mp.Queue()
                proc = mp.Process(target=tracker_process, args=(tracker, tasks, results), name=f"Tracker-{cls}")
                proc.start()
                self.trackers[cls] = (proc, tasks, results)
            self.trackers[cls][1].put(["update", bbs])

        t2 = time.time()

        for cls in per_class.keys():
            tracked[cls] = self.trackers[cls][2].get()

        t3 = time.time()

        (self._bench.measurements()
         .add("pre.process", t1 - t0)
         .add("tracker.process.update", t2 - t1)
         .add("tracker.process.results", t3 - t2)).done()

        return tracked

    def reset(self):
        for _, t, _ in self.trackers.values():
            t.put(["reset", None])

    def cleanup(self):
        for p, q, _ in self.trackers.values():
            q.put(["exit", None])
            p.join()
        self.trackers = {}
