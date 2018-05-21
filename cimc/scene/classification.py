import attr
import imageio
import numpy as np
from tqdm import tqdm

import cimc.models.places.labels as lbl
from cimc import resources, utils
from cimc.models.places.places import Places365, SceneType, CategoryPrediction, PlacesClassification


@attr.s(slots=True)
class SceneClassification(PlacesClassification):
    num_measures: int = attr.ib(default=1)
    length: int = attr.ib(default=1)


class SceneClassifier:
    def __init__(self, fps: float, interval: float = 1):
        self.interval = interval
        self.places_net = Places365.pre_trained().to(utils.best_device)
        self._curr_frame = 0
        self._next_frame = 0
        self._results = []
        self._probs_cats = np.zeros(len(lbl.CATEGORIES))
        self._probs_acc = np.zeros(len(lbl.ATTRIBUTES))

    def update(self, frame: np.ndarray, idx: int = None):
        if idx is not None:
            self._curr_frame = idx
        pass

    def classify(self):
        pass

    def reset(self):
        pass


def classify_scene(video: imageio.core.Format.Reader, interval: float = 1):
    places_net = Places365.pre_trained()
    places_net.to(utils.best_device)
    fps = video.get_meta_data()['fps']
    step = int(interval * fps)
    next_frame = 0
    results = []
    probs_cats = np.zeros(len(lbl.CATEGORIES))
    attrs_acc = np.zeros(len(lbl.ATTRIBUTES))
    with tqdm(enumerate(video), total=len(video)) as bar:
        for i, frame in bar:
            if i != next_frame:
                continue
            res = places_net.classify(frame)
            idx_cats = [category.id for category in res.categories]
            probs_cats[idx_cats] += [category.confidence for category in res.categories]
            attrs_acc[res.attributes['id']] += 1
            results.append(res)
            next_frame += step

    types = np.array([r.type.value for r in results])
    ty = SceneType(np.round(np.mean(types)))
    idx_top_5_cats = np.argsort(probs_cats)[::-1][:5]
    top_5_cats = lbl.CATEGORIES[idx_top_5_cats]
    conf_top_5_cats = probs_cats[idx_top_5_cats] / len(results)
    categories = [CategoryPrediction(id=cat['id'],
                                     name=cat['label'],
                                     confidence=conf)
                  for cat, conf in zip(top_5_cats, conf_top_5_cats)]
    attributes = lbl.ATTRIBUTES[np.argsort(attrs_acc)[::-1][:10]][['id', 'label']]
    result = SceneClassification(type=ty,
                                 categories=categories,
                                 length=len(video),
                                 num_measures=len(results))
    return result


def test_classify_scene():
    import imageio
    test_video = resources.video('Venice-1.mp4')
    with imageio.get_reader(test_video) as video:
        classify_scene(video)


if __name__ == '__main__':
    test_classify_scene()
