import imageio
import numpy as np
from tqdm import tqdm

from cimc import resources, utils
from cimc.models.places.places import Places365, SceneType


def classify_scene(video: imageio.core.Format.Reader, interval: float = 1):
    places = Places365.pre_trained()
    places.to(utils.best_device)
    fps = video.get_meta_data()['fps']
    step = int(interval * fps)
    next_frame = 0
    results = []
    with tqdm(enumerate(video), total=len(video)) as bar:
        for i, frame in bar:
            if i != next_frame:
                continue
            results.append(places.classify(frame))
            next_frame += step

    types = np.array([r.type.value for r in results])
    ty = SceneType(np.round(np.mean(types)))
    pass


def test_classify_scene():
    import imageio
    test_video = resources.video('Venice-1.mp4')
    with imageio.get_reader(test_video) as video:
        classify_scene(video)


if __name__ == '__main__':
    test_classify_scene()
