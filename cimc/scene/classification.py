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
    probs_cats = np.zeros(len(places.classes))
    with tqdm(enumerate(video), total=len(video)) as bar:
        for i, frame in bar:
            if i != next_frame:
                continue
            res = places.classify(frame)
            idx_cats = [category.id for category in res.categories]
            probs_cats[idx_cats] += [category.confidence for category in res.categories]
            results.append(res)
            next_frame += step

    types = np.array([r.type.value for r in results])
    ty = SceneType(np.round(np.mean(types)))
    idx_top_5_cats = np.argsort(probs_cats)[::-1][:5]
    top_5_cats = places.classes[idx_top_5_cats]
    conf_top_5_cats = probs_cats[idx_top_5_cats] / len(results)
    cat_i = np.argmax(probs_cats)
    category = (cat_i, places.classes[cat_i], probs_cats[cat_i] / len(results))
    pass


def test_classify_scene():
    import imageio
    test_video = resources.video('Venice-1.mp4')
    with imageio.get_reader(test_video) as video:
        classify_scene(video)


if __name__ == '__main__':
    test_classify_scene()
