import time

import cv2
import numba as nb
import numpy as np
import torch
from numba import jit

from cimc import resources

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rgb2hsv(frame):
    arr = frame
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(frame)
    assert isinstance(arr, torch.Tensor)
    arr = arr.type(torch.float32).detach() / 255
    out = torch.empty_like(arr)
    # V channel
    out_v = torch.max(arr, -1)[0]

    # S channel
    delta = out_v - torch.min(arr, -1)[0]
    out_s = delta / out_v
    out_s[delta == 0.] = 0.

    # H channel
    # red is max
    max_r = (arr[:, :, 0] == out_v)
    out[max_r, [0]] = (arr[max_r, [1]] - arr[max_r, [2]]) / delta[max_r]

    # green is max
    max_g = (arr[:, :, 1] == out_v)
    out[max_g, [0]] = 2. + (arr[max_g, [2]] - arr[max_g, [0]]) / delta[max_g]

    # blue is max
    max_b = (arr[:, :, 2] == out_v)
    out[max_b, [0]] = 4. + (arr[max_b, [0]] - arr[max_b, [1]]) / delta[max_b]

    out_h = (out[:, :, 0] / 6.) % 1.
    out_h[delta == 0.] = 0.

    out[:, :, 0] = out_h
    out[:, :, 1] = out_s
    out[:, :, 2] = out_v

    out[torch.isnan(out)] = 0.
    return out


# @jit(nb.boolean(nb.float64[:, :, :], nb.float32))
def is_cut(frames: np.ndarray, threshold: float = 30.0) -> bool:
    assert frames.shape[0] >= 2, 'Needs at least 2 frames'
    frame_1, frame_2 = frames[0], frames[-1]  # first and last fame
    assert frame_1.shape == frame_2.shape, "Frames need to be the same size"
    frame_1_hsv = cv2.cvtColor(frame_1, cv2.COLOR_RGB2HSV).astype(np.int32)
    frame_2_hsv = cv2.cvtColor(frame_2, cv2.COLOR_RGB2HSV).astype(np.int32)

    delta_hsv = np.mean(np.mean(np.abs(frame_2_hsv - frame_1_hsv), axis=0), axis=0)
    delta_hsv_avg = np.sum(delta_hsv) / 3.0

    return delta_hsv_avg >= threshold


def is_cut_tensor(frames: np.ndarray, threshold: float = 30.0) -> bool:
    assert frames.shape[0] >= 2, 'Needs at least 2 frames'
    frame_1, frame_2 = frames[0], frames[-1]  # first and last fame
    assert frame_1.shape == frame_2.shape, "Frames need to be the same size"
    frame_1, frame_2 = torch.from_numpy(frame_1).to(device), \
                       torch.from_numpy(frame_2).to(device)
    frame_1_hsv = rgb2hsv(frame_1)
    frame_2_hsv = rgb2hsv(frame_2)

    frame_1_hsv[:, :, 0] *= 180
    frame_2_hsv[:, :, 0] *= 180

    frame_1_hsv[:, :, [1, 2]] *= 255
    frame_2_hsv[:, :, [1, 2]] *= 255

    delta_hsv = torch.mean(torch.mean(torch.abs(frame_2_hsv - frame_1_hsv), dim=0), dim=0)
    delta_hsv_avg = torch.sum(delta_hsv) / 3.0

    return delta_hsv_avg >= threshold


class SceneDetector:
    def __init__(self, threshold=30.0, min_length=15, downscale=1, skip=1):
        assert 0 < threshold < 100
        assert min_length > 0
        assert downscale >= 1
        self.threshold = threshold
        self.min_length = min_length
        self.skip = skip
        self.downscale_factor = downscale
        self._curr_id = 0
        self._last_frame = None
        self._last_scene = 0
        self._skip_to = 0
        self._timings = []

    def __call__(self, frame: np.ndarray):
        return self.update(frame)

    def update(self, frame: np.ndarray) -> bool:
        cut = False
        if self._curr_id == self._skip_to:
            self._skip_to = self._curr_id + self.skip
            if self._curr_id == 0:
                cut = True
                self._last_scene = 0
            elif (self._last_frame is not None
                  and self._curr_id - self._last_scene >= self.min_length):
                f1 = self.downscale(self._last_frame, self.downscale_factor)
                f2 = self.downscale(frame, self.downscale_factor)
                t1 = time.time()
                cut = is_cut(np.array([f1, f2]), self.threshold)
                self._timings.append(time.time() - t1)
            self._last_frame = frame
        self._curr_id += 1
        return cut

    @staticmethod
    def downscale(frame: np.ndarray, factor: int = 1) -> np.ndarray:
        return frame[::factor, ::factor, :]


def detect(video_uri: str):
    import imageio
    import os.path
    from tqdm import tqdm
    detector = SceneDetector(downscale=4, skip=1)
    scenes = []
    bleep_cut = 0
    with imageio.get_reader(video_uri) as video:
        timings = np.zeros(len(video), np.double)
        fps = video.get_meta_data()['fps']
        out_uri, ext = os.path.splitext(video_uri)
        out_uri = f"{out_uri}-scenes.{ext}"
        with imageio.get_writer(out_uri,
                                fps=fps,
                                quality=5) as out:
            bar = tqdm(total=len(video))
            for i, frame in enumerate(video):
                bar.update()
                t1 = time.time()
                if detector.update(frame):
                    bleep_cut += int(fps / 4)
                    bar.set_postfix(last_scene=i)
                    scenes.append(i)
                timings[i] = time.time() - t1
                out_frame = frame
                if bleep_cut > 0:
                    out_frame[-100:, -100:] = [255, 0, 0]
                    bleep_cut -= 1
                out.append_data(out_frame)
            bar.close()
        if scenes[-1] != len(video):
            scenes.append(len(video))
    print(f"Avg cut detection time: {np.mean(np.array(detector._timings))*1e3:.3f}ms")
    print(f"Avg detector update time: {np.mean(timings)*1e3:.3f}ms")
    return scenes


def test_rgb2hsv():
    ex_frame = torch.tensor([[[255, 0, 0], [0, 255, 0]],
                             [[0, 0, 255], [200, 200, 50]]],
                            dtype=torch.double)
    rgb = rgb2hsv(ex_frame)


def test_detector():
    sample = np.ones((10, 10, 3), np.uint8)
    scene_1 = (sample * [255, 0, 0]).astype(np.uint8)
    scene_2 = (sample * [0, 255, 0]).astype(np.uint8)
    scene_3 = (sample * [0, 0, 255]).astype(np.uint8)
    detector = SceneDetector(threshold=5)
    assert detector.update(scene_1), "First frame should always be a new scene"
    for _ in range(detector.min_length - 1):
        assert not detector.update(scene_2), "No scene cuts if in min length"
    assert detector._curr_id == detector.min_length
    assert detector.update(scene_3), "Should detect after min length"

    # With a skip of 3 and min_length of 10 the first cut should be found
    # on frame number 12 (Saved frames: 0, 3, 6, 9, 12, 15, ...)
    detector = SceneDetector(threshold=5, min_length=10, skip=3)
    for _ in range(detector.min_length):
        detector.update(scene_1)
    assert not detector.update(scene_2), "Should skip frames"
    detector.update(scene_2)
    assert detector._curr_id == 12
    assert detector.update(scene_2)


def test_detect():
    scenes = detect(resources.video('goldeneye.mp4'))
    print(scenes)
    scene_ranges = list(zip(scenes[:-1], (scenes + [0])[1:]))
    print(scene_ranges)


if __name__ == '__main__':
    test_detect()
