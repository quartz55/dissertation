import argparse
import logging
import os
from typing import Optional
from zipfile import ZipFile

import aiohttp
import attr
import numpy as np
import pandas as pd
import sklearn.metrics
from aiostream import stream
from plotnine import *

from cimc import utils
from cimc.classifier.utils import get_clsf
from cimc.similarity import video_similarity
from cimc.utils import log, downloader

logger = logging.getLogger(__name__)
logger.addHandler(log.TqdmLoggingHandler())
logger.setLevel(logging.DEBUG)


def rel_path(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), path)


def path_for(video, base_dir: str):
    return os.path.join(base_dir, str(video["query_id"]), video["video_name"])


class CCWebVideoDataset:
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            root_dir = rel_path("cc_web_dataset")
        elif os.path.exists(root_dir) and os.path.isfile(root_dir):
            raise FileExistsError(f"Can't use directory '{root_dir}' because it is a file")

        self.root_dir: str = root_dir
        self.videos: Optional[pd.DataFrame] = None
        self.seeds: Optional[pd.DataFrame] = None
        self.shots: Optional[pd.DataFrame] = None

    def run_for_query(self, query_id: int, sim_thres: float = 0.65):
        vid_dir = os.path.join(self.root_dir, "videos")
        seed_video_id = int(self.seeds.loc[str(query_id)]["seed_video_id"])
        seed_video = self.videos.loc[seed_video_id]
        seed_clsf = get_clsf(video_uri=path_for(seed_video, base_dir=vid_dir))
        gt = pd.read_csv(
            os.path.join(self.root_dir, "GT", f"GT_{query_id}.rst"),
            sep="\t",
            header=None,
            names=["video_id", "status"],
            index_col=0,
        )

        results = []
        for video_id, truth in gt.iterrows():
            video = self.videos.loc[video_id]

            try:
                clsf = get_clsf(video_uri=path_for(video, base_dir=vid_dir), force=False)
            except (FileNotFoundError, ValueError):
                continue
            status = truth["status"]
            actual_similar = status in ['E', 'S', 'V', 'L']

            sim_res = video_similarity(seed_clsf, clsf, threshold=sim_thres)
            sim_val = np.mean(np.array([a[0][1] if len(a) > 0 else 0
                                        for a in sim_res.assignments]))

            results.append({"query_id": query_id,
                            "video_id": video_id,
                            "seed_id": seed_video_id,
                            "status": status,
                            "actual": actual_similar,
                            "similarity": np.float(sim_val)})
        results = pd.DataFrame(results)
        results.set_index(["video_id", "query_id"], inplace=True)
        return results

    def classify_for_query(self, query_id: int):
        vid_dir = os.path.join(self.root_dir, "videos")
        ex = self.videos[self.videos["query_id"] == query_id]
        for i, v in ex.iterrows():
            get_clsf(video_uri=path_for(v, base_dir=vid_dir), force=True)

    async def load(self):
        utils.mkdir_p(self.root_dir)
        await self.download_files(self.root_dir)
        self.videos = pd.read_csv(
            os.path.join(self.root_dir, "videos.tsv"),
            sep="\t",
            header=None,
            names=["video_id", "query_id", "source", "video_name", "url"],
            index_col=0,
        )
        self.seeds = pd.read_csv(
            os.path.join(self.root_dir, "seeds.tsv"),
            sep="\t",
            header=None,
            names=["query_id", "seed_video_id"],
            index_col=0,
        )
        self.shots = pd.read_csv(
            os.path.join(self.root_dir, "shots.tsv"),
            sep="\t",
            header=None,
            names=["serial_id", "keyframe_name", "video_id", "video_name"],
            index_col=[2, 0],
        )

    async def __aenter__(self):
        await self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def videos_from_query(self, query_id: int):
        v = self.videos
        return v[v["query_id"] == query_id]

    @staticmethod
    async def download_videos(videos: pd.DataFrame, directory=os.getcwd()):
        # http://vireo.cs.cityu.edu.hk/webvideo/videos/QueryID/VideoName
        base_url = "http://vireo.cs.cityu.edu.hk/webvideo/videos"

        async with aiohttp.ClientSession() as session:
            for i, v in videos.iterrows():
                path = path_for(v, directory)
                utils.mkdir_p(path)
                url = f"{base_url}/{v['query_id']}/{v['video_name']}"
                await utils.downloader.download(url, path, segments=1, session=session)

    @staticmethod
    async def download_files(directory=os.getcwd()):
        urls = {
            "videos": "http://vireo.cs.cityu.edu.hk/webvideo/Info/Video_List.txt",
            "shots": "http://vireo.cs.cityu.edu.hk/webvideo/Info/Shot_Info.txt",
            "seeds": "http://vireo.cs.cityu.edu.hk/webvideo/Info/Seed.txt",
            "gt_files": "http://vireo.cs.cityu.edu.hk/webvideo/Info/Ground.zip",
        }

        utils.mkdir_p(directory)

        async def _gt_aux():
            _, path = await utils.downloader.download(urls["gt_files"], "gt.zip", segments=1,
                                                      directory=directory)
            logger.info(f"Extracting ground truth zip file: '{path}'")
            with ZipFile(path) as gt_zip:
                gt_zip.extractall(directory)

        paths = [(urls["videos"], "videos.tsv"), (urls["shots"], "shots.tsv"), (urls["seeds"], "seeds.tsv")]
        others = utils.downloader.download_queue(paths, directory=directory, segments=1, parallel=True)

        await stream.merge(stream.just(others), stream.just(_gt_aux()))


async def prepare_query(query_id: int):
    async with CCWebVideoDataset() as ds:
        await ds.download_videos(ds.videos_from_query(query_id), os.path.join(ds.root_dir, "videos"))
        return ds


def conf_matrix_plot(conf_df):
    text_color = np.array(["black"] * len(conf_df))
    text_color[conf_df["entries"] < np.max(conf_df["entries"]) / 2] = "white"
    return (ggplot(conf_df, aes(x="actual", y="predicted", fill="entries")) +
            geom_tile() +
            scale_fill_gradient() +
            geom_text(aes(label="entries"), size=32, color=text_color) +
            labs(title="Confusion Matrix",
                 x="Actual",
                 y="Predicted"))


def confusion_matrix(predictions: pd.DataFrame, dataframe=False):
    conf_matrix = sklearn.metrics.confusion_matrix(predictions["actual"], predictions["predicted"])
    if dataframe:
        return pd.DataFrame({"actual": ["false", "false", "true", "true"],
                             "predicted": ["false", "true", "false", "true"],
                             "entries": np.hstack(conf_matrix)})
    return conf_matrix


@attr.s(slots=True, frozen=True)
class Report:
    precision: float = attr.ib()
    recall: float = attr.ib()
    F1: float = attr.ib()
    TP: int = attr.ib()
    TN: int = attr.ib()
    FP: int = attr.ib()
    FN: int = attr.ib()


def prediction_report(predictions):
    conf_mat = confusion_matrix(predictions)
    TP = conf_mat[1][1]
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]

    total_pos = TP + FP
    tp_fn = TP + FN

    precision = 0 if total_pos == 0 else TP / total_pos
    recall = 0 if tp_fn == 0 else TP / tp_fn
    F1 = 2 * (precision * recall) / (precision + recall)
    return Report(precision, recall, F1,
                  TP, TN, FP, FN)


def calculate_roc_curve(results):
    fpr, tpr, thres = sklearn.metrics.roc_curve(results["actual"], results["similarity"])
    roc = pd.DataFrame({"threshold": thres,
                        "fpr": fpr,
                        "tpr": tpr})
    return roc


def roc_curve_plot(measures, points=True):
    plot = (ggplot(measures, aes(x="fpr", y="tpr")) +
            geom_line(size=2, alpha=0.7, color="orange") +
            labs(title="ROC Curve",
                 x="False Positive Rate",
                 y="True Positive Rate"))
    if points:
        plot += geom_point(alpha=0.35, stroke=0)
    return plot


def get_best_thres(roc):
    aux = roc.assign(max=roc["tpr"] - roc["fpr"])
    return aux.loc[aux["max"].idxmax()]


async def grab_results(queries):
    acc = {}
    for query_id in queries:
        async with CCWebVideoDataset() as ds:
            results = ds.run_for_query(query_id, sim_thres=0.65)
            results.to_csv(f"query_{query_id}.results.csv")
            acc[query_id] = results
    return acc


QUERIES_DONE = [1, 2, 7, 13, 14, 16, 20]


def cell1():
    res = pd.DataFrame()
    for i in QUERIES_DONE:
        res = res.append(pd.read_csv(f"query_{i}.results.csv"))
    roc = calculate_roc_curve(res)
    best_thres = get_best_thres(roc)
    thres = best_thres["threshold"]
    preds = res.assign(predicted=res["similarity"] >= thres)
    conf_df = confusion_matrix(preds, dataframe=True)

    roc_plot = (roc_curve_plot(roc, points=False) +
                ggtitle(f"ROC Curve"))
    (roc_plot + theme_538()).save(f"all_538.roc.jpg")
    (roc_plot + theme_seaborn()).save(f"all_seaborn.roc.jpg")

    conf_plot = (conf_matrix_plot(conf_df) +
                 ggtitle(f"Confusion Matrix (thres={thres:.2f})"))
    (conf_plot + theme_538()).save(f"all_538.conf.jpg")
    (conf_plot + theme_seaborn()).save(f"all_seaborn.conf.jpg")


def cell2():
    for i in QUERIES_DONE:
        res = pd.read_csv(f"query_{i}.results.csv")
        roc = calculate_roc_curve(res)
        best_thres = get_best_thres(roc)
        thres = best_thres["threshold"]
        print(best_thres, thres)
        preds = res.assign(predicted=res["similarity"] >= thres)
        conf_df = confusion_matrix(preds, dataframe=True)

        roc_plot = (roc_curve_plot(roc) +
                    ggtitle(f"ROC Curve (query={i})"))
        (roc_plot + theme_538()).save(f"query_{i}_538.roc.jpg")
        (roc_plot + theme_seaborn()).save(f"query_{i}_seaborn.roc.jpg")

        conf_plot = (conf_matrix_plot(conf_df) +
                     ggtitle(f"Confusion Matrix (query={i}, thres={thres:.2f})"))
        (conf_plot + theme_538()).save(f"query_{i}_538.conf.jpg")
        (conf_plot + theme_seaborn()).save(f"query_{i}_seaborn.conf.jpg")


def workbook():
    # utils.run_future_sync(grab_results([1, 2, 4, 7, 13, 14, 20]))

    parser = argparse.ArgumentParser(description='CC_WEB_VIDEO dataset script')
    parser.add_argument('task', type=str, choices=['classify', 'ndvd'],
                        help='Task to run [classify,ndvd]')
    parser.add_argument('query_id', type=int, choices=range(1, 25),
                        help='QueryID from 1-24')
    args = parser.parse_args()
    query_id = args.query_id
    ds: CCWebVideoDataset = utils.run_future_sync(prepare_query(query_id))
    print(args)
    if args.task == 'classify':
        ds.classify_for_query(query_id)
    elif args.task == 'ndvd':
        results = ds.run_for_query(query_id, sim_thres=0.65)
        pred_65 = results.assign(predicted=results["similarity"] >= 0.65)
        conf_matrix_65 = confusion_matrix(results, 0.65)
        conf_df_65 = confusion_matrix(results, 0.65, dataframe=True)
        roc = calculate_roc_curve(results)
        roc_curve_plot(roc).draw().show()
        pass


if __name__ == "__main__":
    workbook()
