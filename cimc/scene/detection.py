import scenedetect as sd
from scenedetect.detectors import ContentDetector
from scenedetect.manager import SceneManager


class SceneDetector:
    def __init__(self, threshold=0.3, min_length=15, downscale=1):
        self.downscale = downscale
        self._detector = ContentDetector(threshold=threshold, min_scene_len=min_length)

    def __call__(self, video_uri: str):
        smgr = SceneManager(detector=self._detector,
                            downscale_factor=self.downscale)
        sd.detect_scenes_file(video_uri, smgr)
        return smgr.scene_list


if __name__ == '__main__':
    detector = SceneDetector(downscale=4)
    scenes = detector('resources/videos/bvs.mp4')
    print(scenes)
