import pickle
from typing import Set, Dict, List

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as metrics

import cimc.models.places.labels as places_labels
import cimc.models.yolov3.labels as yolo_labels
from cimc import resources
from cimc.classifier import Segment, VideoClassification
from cimc.models.places import SceneType

TYPE = (0, 1)
DURATION = (sum(TYPE), 1)
CATEGORIES = (sum(DURATION), len(places_labels.CATEGORIES))
ATTRIBUTES = (sum(CATEGORIES), len(places_labels.ATTRIBUTES))
OBJECTS = (sum(ATTRIBUTES), len(yolo_labels.COCO_LABELS))
COLUMNS = sum(
    map(lambda x: x[1], [TYPE, DURATION, CATEGORIES, ATTRIBUTES, OBJECTS]))
HEADERS = ['Type'] + \
          ['Duration'] + \
          [f'sc_{name}' for name in places_labels.CATEGORIES['label']] + \
          [f'attr_{name}' for name in places_labels.ATTRIBUTES['label']] + \
          [f'cls_{name}' for name in yolo_labels.COCO_LABELS]


def _obj_id_by_class(segment: Segment):
    classes: Dict[int, Set[int]] = {}
    for objects in segment.objects:
        for cls, cls_objs in objects.items():
            if cls not in classes:
                classes[cls] = set()
            for obj in cls_objs:
                classes[cls].add(obj.tracking_id)
    return classes


def clean_vector(v: pd.DataFrame):
    return v.loc[:, (v != 0).any(axis=0)]


def make_feature_vector(segment: Segment):
    fv = np.zeros(COLUMNS, dtype=np.double)
    objs_by_class = _obj_id_by_class(segment)
    # Type
    s_type = segment.scene.type
    fv[TYPE[0]] = 1 if s_type is SceneType.INDOOR else 0
    # fv[DURATION[0]] = segment.duration
    fv[DURATION[0]] = 0

    # Categories
    categories = segment.scene.categories
    conf_rest = 1 - sum([c.confidence for c in categories])
    conf_add = conf_rest / len(categories)
    for cat in categories:
        fv[CATEGORIES[0] + cat.id] = cat.confidence + conf_add

    # Attributes
    for attr in segment.scene.attributes:
        fv[ATTRIBUTES[0] + attr.id] = 1 / len(segment.scene.attributes)

    # Objects
    total = sum([len(o) for o in objs_by_class.values()])
    for cls_id, objs_ids in objs_by_class.items():
        fv[OBJECTS[0] + cls_id] = 0 if total == 0 else len(objs_ids) / total

    return pd.DataFrame(fv[np.newaxis], columns=HEADERS, index=[segment.id])


def _aux(classes):
    classes_label = {yolo_labels.COCO_LABELS[id]: {'number': len(objs),
                                                   'set': objs}
                     for id, objs in classes.items()}
    return classes_label


def similarity(segments: List[Segment], other: List[Segment] = None):
    if other is not None and isinstance(other, list) and len(other) > 0:
        fvs = pd.concat([make_feature_vector(s) for s in segments + other])
        fvs_clean = clean_vector(fvs)
        fvs_1, fvs_2 = fvs_clean.iloc[:len(segments)], fvs_clean.iloc[len(segments):]
        return metrics.cosine_similarity(fvs_1, fvs_2)
        # return metrics.euclidean_distances(fvs_1, fvs_2)
    else:
        fvs = pd.concat([make_feature_vector(s) for s in segments])
        fvs_clean = clean_vector(fvs)
        return metrics.cosine_similarity(fvs_clean)
        # return metrics.euclidean_distances(fvs_clean)


def _timings():
    import timeit
    with open(resources.video("TUD-Campus.mp4.clsf"), 'rb') as fd:
        clsf: VideoClassification = pickle.load(fd)
    segment = clsf.segments[0]
    cls1 = _aux(_obj_id_by_class(segment))
    fv1 = make_feature_vector(segment)
    fv1_raw_clean = fv1.values[:, (fv1.values != 0).any(0)]
    fv1_clean = clean_vector(fv1)

    with open(resources.video("TUD-Campus.var.rotate-scale.mp4.clsf"), 'rb') as fd:
        clsf: VideoClassification = pickle.load(fd)
    segment = clsf.segments[0]
    cls2 = _aux(_obj_id_by_class(segment))
    fv2 = make_feature_vector(segment)
    fv2_raw_clean = fv1.values[:, (fv1.values != 0).any(0)]
    fv2_clean = clean_vector(fv2)

    fvs = pd.concat([fv1, fv2])
    fvs_clean = clean_vector(fvs)

    t1_raw = timeit.timeit("metrics.cosine_similarity(fv1.values, fv2.values)",
                           number=10000,
                           globals={**globals(), **locals()})
    t1_raw_clean = timeit.timeit("metrics.cosine_similarity(fv1_raw_clean, fv2_raw_clean)",
                                 number=10000,
                                 globals={**globals(), **locals()})
    t1_df_single = timeit.timeit("metrics.cosine_similarity(fv1, fv2)",
                                 number=10000, globals={**globals(), **locals()})
    t1_df_single_clean = timeit.timeit("metrics.cosine_similarity(fv1_clean, fv2_clean)",
                                       number=10000,
                                       globals={**globals(), **locals()})
    t1_df = timeit.timeit("metrics.cosine_similarity(fvs)",
                          number=10000,
                          globals={**globals(), **locals()})
    t1_df_clean = timeit.timeit("metrics.cosine_similarity(fvs_clean)",
                                number=10000,
                                globals={**globals(), **locals()})

    timings = \
        f"""
        Raw numpy: {t1_raw:.04f}s ({t1_raw * 1e3 / 10000:.04f}ms p/ iteration)
        Raw numpy (clean): {t1_raw_clean:.04f}s ({t1_raw_clean * 1e3 / 10000:.04f}ms p/ iteration)
        2 Dataframes: {t1_df_single:.04f}s ({t1_df_single * 1e3 / 10000:.04f}ms p/ iteration)
        2 Dataframes (clean): {t1_df_single_clean:.04f}s ({t1_df_single_clean * 1e3 / 10000:.04f}ms p/ iteration)
        Dataframe: {t1_df:.04f}s ({t1_df * 1e3 / 10000:.04f}ms p/ iteration)
        Dataframe (clean): {t1_df_clean:.04f}s ({t1_df_clean * 1e3 / 10000:.04f}ms p/ iteration)
        """
    print(timings)


vid1 = resources.video("TUD-Campus.mp4.clsf")
vid1_alt1 = resources.video("TUD-Campus.var.rotate-scale.mp4.clsf")
vid1_alt2 = resources.video("TUD-Campus.var.rotate-scale-flip-color.mp4.clsf")

vid2 = resources.video("goldeneye.mp4.clsf")
vid2_2x = resources.video("goldeneye-2x.mp4.clsf")

vid3 = resources.video("TUD-Crossing.mp4.clsf")

vid4 = resources.video("justice-league.mp4.clsf")

vid5 = resources.video("goldeneye-justiceleague.mp4.clsf")


def vid_similarity(vid1, vid2):
    with open(vid1, 'rb') as fd1:
        with open(vid2, 'rb') as fd2:
            clsf1: VideoClassification = pickle.load(fd1)
            clsf2: VideoClassification = pickle.load(fd2)
            sim_2 = similarity(clsf1.segments, clsf2.segments)
            print(f"First scene distance: {sim_2[0][0] * 100:.2f}%")
            return sim_2


def main():
    # s1 = vid_similarity(vid1, vid1_alt1)
    # s2 = vid_similarity(vid1, vid1_alt2)
    # s3 = vid_similarity(vid1, vid3)
    # s4 = vid_similarity(vid1, vid2)
    # s5 = vid_similarity(vid2, vid2_2x)
    # s6 = vid_similarity(vid24 vid4)

    s7 = vid_similarity(vid2, vid5)
    s8 = vid_similarity(vid4, vid5)
    pass


if __name__ == '__main__':
    main()
