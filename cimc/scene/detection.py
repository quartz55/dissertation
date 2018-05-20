from typing import Optional, Tuple

import cv2
import numpy as np
import numba as nb
from numba import jit

from cimc import resources


@jit(nb.boolean(nb.float64[:, :, :], nb.float32))
def is_cut(frames: np.ndarray, threshold: float = 30.0) -> bool:
    assert frames.shape[0] >= 2, 'Needs at least 2 frames'
    frame_1, frame_2 = frames[0], frames[-1]  # first and last fame
    assert frame_1.shape == frame_2.shape, "Frames need to be the same size"
    frame_1_hsv = cv2.split(cv2.cvtColor(frame_1, cv2.COLOR_RGB2HSV))
    frame_2_hsv = cv2.split(cv2.cvtColor(frame_2, cv2.COLOR_RGB2HSV))

    delta_hsv = np.array([-1, -1, -1], dtype=np.float64)
    for i in range(3):
        num_pixels = frame_2_hsv[i].shape[0] * frame_2_hsv[i].shape[1]
        frame_1_hsv[i] = frame_1_hsv[i].astype(np.int32)
        frame_2_hsv[i] = frame_2_hsv[i].astype(np.int32)
        delta_hsv[i] = np.sum(np.abs(frame_2_hsv[i] - frame_1_hsv[i])) / float(num_pixels)
    delta_hsv_avg: float = np.sum(delta_hsv) / 3.0

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
                cut = is_cut(np.array([f1, f2]), self.threshold)
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
    detector = SceneDetector(downscale=4, skip=5)
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


if __name__ == '__main__':
    scenes = detect(resources.video('goldeneye.mp4'))
    print(scenes)
    scene_ranges = list(zip(scenes[:-1], (scenes + [0])[1:]))
    print(scene_ranges)
