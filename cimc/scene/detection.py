import time

import cv2
import numba as nb
import numpy as np

from cimc import resources
from cimc.utils import bench


def rgb_2_hsv(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.int32)


@nb.njit(nb.boolean(nb.int32[:, :, :], nb.int32[:, :, :], nb.float32))
def is_cut(a: np.ndarray, b: np.ndarray, threshold: float = 30.0) -> bool:
    # assert a.shape == b.shape, "Frames need to be the same size"
    if a.shape != b.shape:
        raise ValueError("Frames need to be the same size")

    diff = np.abs(a - b)

    delta_hsv = np.empty(3, np.float32)
    for i in range(3):
        delta_hsv[i] = np.mean(diff[:, :, i])

    delta_hsv_avg = np.sum(delta_hsv) / 3.0

    return np.bool(delta_hsv_avg >= threshold)


@nb.njit(nb.uint8[:, :, :](nb.uint8[:, :, :], nb.uint8))
def downscale(frame: np.ndarray, factor: int) -> np.ndarray:
    return frame[::factor, ::factor, :]


def frame_grabber(skip=0):
    curr = 0
    next = 0
    while True:
        ret = None
        if next == 0:
            ret = curr
            next = skip
        else:
            next -= 1
        curr += 1
        yield ret


class SceneDetector:
    def __init__(self, threshold=30.0, min_length=15, downscale=1, skip=0):
        assert 0 < threshold < 100
        assert min_length > 0
        assert downscale >= 1
        self.threshold = threshold
        self.min_length = min_length
        self.skip = skip
        self.downscale_factor = downscale
        self._curr_id = 0
        self._last_frame = None
        self._last_processed = None
        self._last_scene = 0
        self._skip_to = 0
        self._grabber = frame_grabber(skip)
        self._bench = bench.Bench("shot.detect")

    def __call__(self, frame: np.ndarray):
        return self.update(frame)

    def update(self, frame: np.ndarray) -> bool:
        measurements = self._bench.measurements()
        t0 = time.time()

        cut = False
        frame_id = next(self._grabber)
        if frame_id is not None:
            if frame_id == 0:
                cut = True

            if frame_id - self._last_scene >= self.min_length:
                t1 = time.time()

                if self._last_processed is None:
                    self._last_processed = rgb_2_hsv(downscale(self._last_frame, self.downscale_factor))

                f1 = self._last_processed
                f2 = rgb_2_hsv(downscale(frame, self.downscale_factor))

                t2 = time.time()

                cut = is_cut(f1, f2, self.threshold)

                t3 = time.time()

                if cut:
                    self._last_scene = frame_id

                self._last_processed = f2

                (measurements
                 .add("pre.process", t2 - t1)
                 .add("hsv.comp", t3 - t2)
                 .add("cut.detect", t3 - t1))
            else:
                self._last_processed = None
                self._last_frame = frame

            if cut:
                self._last_scene = frame_id

        measurements.add("iteration", time.time() - t0)
        measurements.done()

        return cut

        # cut = False
        # if self._curr_id == self._skip_to:
        #     self._skip_to = self._curr_id + self.skip
        #
        #     if self._curr_id == 0:
        #         cut = True
        #         self._last_scene = 0
        #
        #     elif (self._last_frame is not None
        #           and self._curr_id - self._last_scene >= self.min_length):
        #
        #         t1 = time.time()
        #
        #         if self._last_processed is None:
        #             down = downscale(self._last_frame, self.downscale_factor)
        #             self._last_processed = rgb_2_hsv(down)
        #
        #         f1 = self._last_processed
        #         f2 = rgb_2_hsv(downscale(frame, self.downscale_factor))
        #
        #         t2 = time.time()
        #
        #         cut = is_cut(f1, f2, self.threshold)
        #         if cut:
        #             self._last_scene = self._curr_id
        #
        #         t3 = time.time()
        #         self._last_processed = f2
        #         (measurements
        #          .add("pre.process", t2 - t1)
        #          .add("hsv.comp", t3 - t2)
        #          .add("cut.detect", t3 - t1))
        #
        #     self._last_frame = frame
        #
        # self._curr_id += 1
        #
        # measurements.add("iteration", time.time() - t0)
        # measurements.done()
        #
        # return cut


def detect(video_uri: str):
    import imageio
    import os.path
    from tqdm import tqdm
    detector = SceneDetector(downscale=4, skip=1)
    scenes = []
    bleep_cut = 0
    with imageio.get_reader(video_uri) as video:
        fps = video.get_meta_data()['fps']
        out_uri, ext = os.path.splitext(video_uri)
        out_uri = f"{out_uri}-scenes.{ext}"
        with imageio.get_writer(out_uri,
                                fps=fps,
                                quality=5) as out:
            bar = tqdm(total=len(video))
            for i, frame in enumerate(video):
                bar.update()
                if detector.update(frame):
                    bleep_cut += int(fps / 4)
                    bar.set_postfix(last_scene=i)
                    scenes.append(i)
                out_frame = frame
                if bleep_cut > 0:
                    out_frame[-100:, -100:] = [255, 0, 0]
                    bleep_cut -= 1
                out.append_data(out_frame)
            bar.close()
        if scenes[-1] != len(video):
            scenes.append(len(video))
    return scenes


def detect2(video_uri: str):
    import imageio
    from tqdm import tqdm
    detector = SceneDetector(downscale=4, skip=1)
    scenes = []
    with imageio.get_reader(video_uri) as video:
        fps = video.get_meta_data()['fps']
        bar = tqdm(total=len(video))
        for i, frame in enumerate(video):
            bar.update()
            if detector.update(frame):
                bar.set_postfix(last_scene=i)
                scenes.append(i)
        bar.close()
    if scenes[-1] != len(video):
        scenes.append(len(video))
    return scenes


def test_detector():
    from itertools import cycle
    sample = np.ones((10, 10, 3), np.uint8)
    scene_1 = (sample * [255, 0, 0]).astype(np.uint8)
    scene_2 = (sample * [0, 255, 0]).astype(np.uint8)
    scene_3 = (sample * [0, 0, 255]).astype(np.uint8)
    detector = SceneDetector(threshold=5)
    assert detector.update(scene_1), "First frame should always be a new scene"
    for _, scene in zip(range(detector.min_length - 1), cycle([scene_2, scene_1])):
        assert detector.update(scene), "No scene cuts if in min length"
    assert detector._curr_id == detector.min_length
    assert detector.update(scene_3), "Should detect after min length"

    # With a skip of 3 and min_length of 10 the first cut should be found
    # on frame number 12 (Saved frames: 0, 3, 6, 9, 12, 15, ...)
    detector = SceneDetector(threshold=5, min_length=10, skip=2)
    for _ in range(detector.min_length):
        detector.update(scene_1)
    assert not detector.update(scene_2), "Should skip frames"
    detector.update(scene_2)
    assert detector._curr_id == 12
    assert detector.update(scene_2)


def test_detect():
    scenes = detect2(resources.video('goldeneye-2x.mp4'))
    scenes = detect2(resources.video('goldeneye-justiceleague.mp4'))
    scenes = detect2(resources.video('TUD-Crossing.mp4'))
    scenes = detect2(resources.video('Venice-1.mp4'))
    # scenes = detect2(resources.video('goldeneye.mp4'))
    # scenes = detect2(resources.video('bvs.mp4'))
    # scenes = detect2(resources.video('justice-league.mp4'))
    print(len(scenes), scenes)
    scene_ranges = list(zip(scenes[:-1], (scenes + [0])[1:]))
    print(scene_ranges)


if __name__ == '__main__':
    test_detect()
