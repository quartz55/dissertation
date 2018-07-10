import logging
import os
import pickle
import time
from typing import Set, Dict, List, Tuple, Optional, Union

import attr
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as metrics

import cimc.models.places.labels as places_labels
import cimc.models.yolov3.labels as yolo_labels
from cimc import resources
from cimc.classifier import Segment, VideoClassification
from cimc.models.places import SceneType
from cimc.utils import log, bench

logger = logging.getLogger(__name__)
logger.addHandler(log.TqdmLoggingHandler())
logger.setLevel(logging.DEBUG)

_bench_gen = bench.Bench("signature.generation")
_bench_comp = bench.Bench("signature.comparison")

TYPE = (0, 1)
DURATION = (sum(TYPE), 1)
CATEGORIES = (sum(DURATION), len(places_labels.CATEGORIES))
ATTRIBUTES = (sum(CATEGORIES), len(places_labels.ATTRIBUTES))
OBJECTS = (sum(ATTRIBUTES), len(yolo_labels.COCO_LABELS))
COLUMNS = sum(map(lambda x: x[1], [TYPE, DURATION, CATEGORIES, ATTRIBUTES, OBJECTS]))
HEADERS = (
        ["Type"]
        + ["Duration"]
        + [f"sc_{name}" for name in places_labels.CATEGORIES["label"]]
        + [f"attr_{name}" for name in places_labels.ATTRIBUTES["label"]]
        + [f"cls_{name}" for name in yolo_labels.COCO_LABELS]
)


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
    m = _bench_gen.measurements()

    t0 = time.time()

    fv = np.zeros(COLUMNS, dtype=np.double)

    # Type
    s_type = segment.scene.type
    fv[TYPE[0]] = 1 if s_type is SceneType.INDOOR else -1

    # Duration
    # fv[DURATION[0]] = segment.duration
    fv[DURATION[0]] = 0

    t1 = time.time()

    # Categories
    cats = segment.scene.categories
    confs = cats["conf"]
    fv[cats["id"] + CATEGORIES[0]] = confs / np.sqrt(confs.dot(confs))

    t2 = time.time()

    # Attributes
    attrs = segment.scene.attributes
    freqs = attrs["freq"]
    fv[attrs["id"] + ATTRIBUTES[0]] = freqs / np.sqrt(freqs.dot(freqs))

    t3 = time.time()

    # Objects
    objs_by_class = _obj_id_by_class(segment)
    objs = np.array(
        [(cls_id, len(objs_ids)) for cls_id, objs_ids in objs_by_class.items()], dtype=[("id", "u4"), ("count", "u4")]
    )
    counts = objs["count"]
    if counts.sum() != 0:
        fv[objs["id"] + OBJECTS[0]] = counts / np.sqrt(counts.dot(counts))

    t4 = time.time()

    # Unit vector
    fv /= np.sqrt(fv.dot(fv))

    t5 = time.time()

    (m.add("fs.categories", t2 - t1)
     .add("fs.attributes", t3 - t2)
     .add("fs.objects", t4 - t3)
     .add("total", t5 - t0)).done()

    return pd.DataFrame(fv[np.newaxis], columns=HEADERS, index=[segment.id])


def _aux(classes):
    classes_label = {yolo_labels.COCO_LABELS[id]: {"number": len(objs), "set": objs} for id, objs in classes.items()}
    return classes_label


def segment_similarity(segments: List[Segment], other: List[Segment] = None):
    if other is not None and isinstance(other, list) and len(other) > 0:
        fvs = pd.concat([make_feature_vector(s) for s in segments + other])
        t0 = time.time()

        fvs_clean = clean_vector(fvs)
        fvs_1, fvs_2 = fvs_clean.iloc[: len(segments)], fvs_clean.iloc[len(segments):]

        t1 = time.time()

        res = metrics.cosine_similarity(fvs_1, fvs_2)

        t2 = time.time()

        _bench_comp.measurement("clean", t1 - t0)
        _bench_comp.measurement("cos.sim", t2 - t1)
        return res
        # return metrics.euclidean_distances(fvs_1, fvs_2)
    else:
        fvs = pd.concat([make_feature_vector(s) for s in segments])
        fvs_clean = clean_vector(fvs)
        res = metrics.cosine_similarity(fvs_clean)
        return res
        # return metrics.euclidean_distances(fvs_clean)


def _timings():
    import timeit

    with open(resources.video("TUD-Campus.mp4.clsf"), "rb") as fd:
        clsf: VideoClassification = pickle.load(fd)
    segment = clsf.segments[0]
    cls1 = _aux(_obj_id_by_class(segment))
    fv1 = make_feature_vector(segment)
    fv1_raw_clean = fv1.values[:, (fv1.values != 0).any(0)]
    fv1_clean = clean_vector(fv1)

    with open(resources.video("TUD-Campus.var.rotate-scale.mp4.clsf"), "rb") as fd:
        clsf: VideoClassification = pickle.load(fd)
    segment = clsf.segments[0]
    cls2 = _aux(_obj_id_by_class(segment))
    fv2 = make_feature_vector(segment)
    fv2_raw_clean = fv1.values[:, (fv1.values != 0).any(0)]
    fv2_clean = clean_vector(fv2)

    fvs = pd.concat([fv1, fv2])
    fvs_clean = clean_vector(fvs)

    t1_raw = timeit.timeit(
        "metrics.cosine_similarity(fv1.values, fv2.values)", number=10000, globals={**globals(), **locals()}
    )
    t1_raw_clean = timeit.timeit(
        "metrics.cosine_similarity(fv1_raw_clean, fv2_raw_clean)", number=10000, globals={**globals(), **locals()}
    )
    t1_df_single = timeit.timeit("metrics.cosine_similarity(fv1, fv2)", number=10000, globals={**globals(), **locals()})
    t1_df_single_clean = timeit.timeit(
        "metrics.cosine_similarity(fv1_clean, fv2_clean)", number=10000, globals={**globals(), **locals()}
    )
    t1_df = timeit.timeit("metrics.cosine_similarity(fvs)", number=10000, globals={**globals(), **locals()})
    t1_df_clean = timeit.timeit("metrics.cosine_similarity(fvs_clean)", number=10000, globals={**globals(), **locals()})

    timings = f"""
        Raw numpy: {t1_raw:.04f}s ({t1_raw * 1e3 / 10000:.04f}ms p/ iteration)
        Raw numpy (clean): {t1_raw_clean:.04f}s ({t1_raw_clean * 1e3 / 10000:.04f}ms p/ iteration)
        2 Dataframes: {t1_df_single:.04f}s ({t1_df_single * 1e3 / 10000:.04f}ms p/ iteration)
        2 Dataframes (clean): {t1_df_single_clean:.04f}s ({t1_df_single_clean * 1e3 / 10000:.04f}ms p/ iteration)
        Dataframe: {t1_df:.04f}s ({t1_df * 1e3 / 10000:.04f}ms p/ iteration)
        Dataframe (clean): {t1_df_clean:.04f}s ({t1_df_clean * 1e3 / 10000:.04f}ms p/ iteration)
        """
    print(timings)


def match_segments(sim_mat: np.ndarray, thres: float = 0.5):
    match_idx = np.argwhere(sim_mat > thres)
    return match_idx


@attr.s(slots=True, frozen=True, repr=False)
class SimilarityResult:
    video_1: Tuple[str, str] = attr.ib()
    video_2: Tuple[str, str] = attr.ib()
    threshold: float = attr.ib()
    assignments: List[List[Tuple[int, float]]] = attr.ib()
    similarity_matrix: Optional[np.ndarray] = attr.ib()

    def __repr__(self):
        aux = f"#### SIMILARITY ####\n" f"# Video1: '{self.video_1[0]}' File: {self.video_1[1]}\n" f"# Video2: '{self.video_2[0]}' File: {self.video_2[1]}\n" f"# Threshold: {self.threshold*100:.2f}%\n" f"#-----------\n" f"# {'Video1 Scene':>12} -> Video2 Matches (showing best 3)\n"
        for sc_id, ass in enumerate(self.assignments):
            ass_str = " ".join(("{:12}".format(f"{id}({sim*100:.2f}%)") for id, sim in ass[:3]))
            divisor = "->" if len(ass_str) > 0 else "||"
            aux += f"# {sc_id:12} {divisor} {ass_str}\n"
        aux += "####################"
        return aux


ClsfArg = Union[str, VideoClassification]


def video_similarity(clsf1_uri: ClsfArg,
                     clsf2_uri: ClsfArg,
                     threshold: float = 0.7):
    clsf1: VideoClassification
    clsf2: VideoClassification
    m = _bench_comp.measurements()

    if isinstance(clsf1_uri, str):
        with open(clsf1_uri, "rb") as fd1:
            clsf1 = pickle.load(fd1)
    elif isinstance(clsf1_uri, VideoClassification):
        clsf1 = clsf1_uri
    else:
        raise ValueError("clsf1 must be a path or a VideoClassification")

    if isinstance(clsf2_uri, str):
        with open(clsf2_uri, "rb") as fd2:
            clsf2 = pickle.load(fd2)
    elif isinstance(clsf2_uri, VideoClassification):
        clsf2 = clsf2_uri
    else:
        raise ValueError("clsf2 must be a path or a VideoClassification")

    sim_matrix = segment_similarity(clsf1.segments, clsf2.segments)

    t0 = time.time()

    matches = match_segments(sim_matrix, threshold)

    t1 = time.time()

    res = [None] * len(sim_matrix)
    for i in range(len(res)):
        matching = matches[:, 1][matches[:, 0] == i]
        similarity = sim_matrix[i][matching]
        sorted_sim_idx = np.argsort(similarity)[::-1]
        seg_res = list(zip(matching[sorted_sim_idx], similarity[sorted_sim_idx]))
        res[i] = seg_res

    t_last = time.time()

    (m
     .add("threshold.filter", t1 - t0)
     .add("segment.matching", t_last - t1)).done()

    return SimilarityResult(
        video_1=(clsf1.name, clsf1.filename),
        video_2=(clsf2.name, clsf2.filename),
        threshold=threshold,
        assignments=res,
        similarity_matrix=sim_matrix,
    )


def test_dataset():
    dataset = resources.load_similarities()
    for name, annotation in dataset.items():
        vid1, vid2 = annotation["video_1"], annotation["video_2"]
        clsf1_uri = resources.video(f"{vid1}.clsf")
        if not os.path.isfile(clsf1_uri):
            logger.error(f"No classification for video {vid1} found: '{clsf1_uri}' not found")
            raise FileNotFoundError(clsf1_uri)

        clsf2_uri = resources.video(f"{vid2}.clsf")
        if not os.path.isfile(clsf2_uri):
            logger.error(f"No classification for video {vid2} found: '{clsf2_uri}' not found")
            raise FileNotFoundError(clsf2_uri)

        calculated = video_similarity(clsf1_uri, clsf2_uri)
        actual = annotation["matches"]
        top_1_acc = 0
        top_3_acc = 0
        stats = {"false_positives": 0}
        for scene_id, matches in enumerate(calculated.assignments):
            true_match = scene_id in actual
            has_matches = len(matches) > 0
            if true_match and has_matches:
                actual_matches = set(actual[scene_id])
                if matches[0][0] in actual_matches:
                    top_1_acc += 1
                    top_3_acc += 1
                elif actual_matches.issubset(set([m[0] for m in matches[:3]])):
                    top_3_acc += 1
            elif not true_match and not has_matches:
                top_1_acc += 1
                top_3_acc += 1
            elif not true_match and has_matches:
                stats["false_positives"] += 1

        total = len(calculated.assignments)
        top_1_percent = top_1_acc / total
        top_3_percent = top_3_acc / total
        logger.info(
            f"""
Annotation: {name}
  Video1: {vid1}
  Video2: {vid2}
  Top 1%: {top_1_percent*100:.2f}%({top_1_acc} / {total})
  Top 3%: {top_3_percent*100:.2f}%({top_3_acc} / {total})"""
        )


def main():
    beach1 = resources.video("beach-1.mp4.clsf")
    beach2 = resources.video("beach-2.mp4.clsf")
    beach_sim = video_similarity(beach1, beach2)

    vid1 = resources.video("TUD-Campus.mp4.clsf")
    vid1_alt1 = resources.video("TUD-Campus.var.rotate-scale.mp4.clsf")
    vid1_alt2 = resources.video("TUD-Campus.var.rotate-scale-flip-color.mp4.clsf")
    vid1_alt3 = resources.video("TUD-Campus.var.vflip-saturated-vignette.mp4.clsf")
    vid1_alt4 = resources.video("TUD-Campus.var.blurred.mp4.clsf")
    vid1_alt5 = resources.video("TUD-Campus.var.textoverlay.mp4.clsf")

    s1 = video_similarity(vid1, vid1_alt1)
    s2 = video_similarity(vid1, vid1_alt2)
    s3 = video_similarity(vid1, vid1_alt3)
    s4 = video_similarity(vid1, vid1_alt4)
    s5 = video_similarity(vid1, vid1_alt5)

    vid2 = resources.video("goldeneye.mp4.clsf")
    vid2_2x = resources.video("goldeneye-2x.mp4.clsf")

    vid3 = resources.video("TUD-Crossing.mp4.clsf")

    vid4 = resources.video("justice-league.mp4.clsf")

    vid5 = resources.video("goldeneye-justiceleague.mp4.clsf")

    # s1 = video_similarity(vid1, vid1_alt1)
    # s2 = video_similarity(vid1, vid1_alt2)
    s3 = video_similarity(vid1, vid3)
    s4 = video_similarity(vid1, vid2)
    s5 = video_similarity(vid2, vid2_2x)
    s6 = video_similarity(vid2, vid4)

    s7 = video_similarity(vid2, vid5)
    s8 = video_similarity(vid4, vid5)
    pass


if __name__ == "__main__":
    main()
    test_dataset()
