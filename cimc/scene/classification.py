import attr
import imageio
import numpy as np
from tqdm import tqdm

import cimc.models.places.labels as lbl
from cimc import resources, utils
from cimc.models.places.places import Places365, SceneType


@attr.s(slots=True, frozen=True)
class CategoryPrediction:
    id: int = attr.ib()
    label: str = attr.ib()
    confidence: float = attr.ib()


@attr.s(slots=True, frozen=True)
class AttributePrediction:
    id: int = attr.ib()
    label: str = attr.ib()
    frequency: float = attr.ib()


@attr.s(slots=True, frozen=True)
class SceneClassification:
    type: SceneType = attr.ib()
    categories: np.ndarray = attr.ib()
    attributes: np.ndarray = attr.ib()
    length: int = attr.ib()
    num_measures: int = attr.ib()


class SceneClassifier:
    def __init__(self, step: int = 1, places_net: Places365 = None):
        if places_net is None:
            places_net = Places365.pre_trained().to(utils.best_device)
        self.places_net = places_net
        self._step = step
        self._next_frame = 0
        self._frames_read = 0
        self._results = []
        self._probs_cats = np.zeros(len(lbl.CATEGORIES))
        self._attrs_acc = np.zeros(len(lbl.ATTRIBUTES))

    def update(self, frame: np.ndarray):
        self._frames_read += 1
        if self._frames_read - 1 < self._next_frame:
            return
        res = self.places_net.classify(frame)
        self._probs_cats[res.categories['id']] += res.categories['confidence']
        self._attrs_acc[res.attributes['id']] += 1
        self._results.append(res)
        self._next_frame += self._step

    def classification(self):
        types = np.array([r.type.value for r in self._results])
        ty = SceneType(np.round(np.mean(types)))

        cats_idx = np.argsort(self._probs_cats)[::-1]
        cats_conf = self._probs_cats[cats_idx] / len(self._results)
        categories = zip(cats_idx, lbl.CATEGORIES[cats_idx]['label'], cats_conf)
        categories = np.fromiter(categories, count=len(lbl.CATEGORIES),
                                 dtype=[('id', 'u4'), ('label', 'U40'), ('conf', 'f4')])

        attrs_idx = np.argsort(self._attrs_acc)[::-1]
        attrs_freq = self._attrs_acc[attrs_idx] / len(self._results)
        attributes = zip(attrs_idx, lbl.ATTRIBUTES[attrs_idx]['label'], attrs_freq)
        attributes = np.fromiter(attributes, count=len(lbl.ATTRIBUTES),
                                 dtype=[('id', 'u4'), ('label', 'U40'), ('freq', 'f4')])

        result = SceneClassification(type=ty,
                                     categories=categories,
                                     attributes=attributes,
                                     length=self._frames_read,
                                     num_measures=len(self._results))
        return result

    def reset(self):
        self._frames_read = 0
        self._next_frame = 0
        self._results = []
        self._probs_cats = np.zeros(len(lbl.CATEGORIES))
        self._attrs_acc = np.zeros(len(lbl.ATTRIBUTES))


def classify_scene(video_uri: str, interval: float = 1):
    with imageio.get_reader(video_uri) as video:
        fps = video.get_meta_data()['fps']
        step = int(interval * fps)
        classifier = SceneClassifier(step=step)
        with tqdm(enumerate(video),
                  total=len(video),
                  desc='Classifying scene',
                  unit='frame') as bar:
            for i, frame in bar:
                classifier.update(frame)
            result = classifier.classification()
            return result


def test_classify_scene():
    result = classify_scene(resources.video('Venice-1.mp4'))
    assert result.type == SceneType.OUTDOOR
    top_5_cats = result.categories[:5]
    top_10_attrs = result.attributes[:10]
    assert 'plaza' in top_5_cats['label']
    assert 'natural light' in top_10_attrs['label']
    assert 'man-made' in top_10_attrs['label']
    assert 'open area' in top_10_attrs['label']


if __name__ == '__main__':
    test_classify_scene()
