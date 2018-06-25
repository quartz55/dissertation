import argparse
import logging
import os
from typing import Optional, Dict
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
from cimc.similarity import video_similarity, SimilarityResult
from cimc.utils import log, downloader

logger = logging.getLogger(__name__)
logger.addHandler(log.TqdmLoggingHandler())
logger.setLevel(logging.DEBUG)


@attr.s(slots=True, frozen=True)
class Result:
    query_id: int = attr.ib()
    video_id: int = attr.ib()
    status: str = attr.ib()
    actual: bool = attr.ib()
    similarity: float = attr.ib()
    _sim_res: Optional[SimilarityResult] = attr.ib()


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
        seed_video_id = self.seeds.loc[str(query_id)]["seed_video_id"]
        seed_video = self.videos.loc[seed_video_id]
        seed_clsf = get_clsf(video_uri=path_for(seed_video, base_dir=vid_dir))
        gt = pd.read_csv(
            os.path.join(self.root_dir, "GT", f"GT_{query_id}.rst"),
            sep="\t",
            header=None,
            names=["video_id", "status"],
            index_col=0,
        )

        results: Dict[int, Result] = {}
        for video_id, truth in gt.iterrows():
            video = self.videos.loc[video_id]

            try:
                clsf = get_clsf(video_uri=path_for(video, base_dir=vid_dir), force=False)
            except ValueError:
                continue

            status = truth["status"]
            actual_similar = status in ['E', 'S', 'V', 'L']

            sim_res = video_similarity(seed_clsf, clsf, threshold=sim_thres)
            sim_val = np.mean(np.array([a[0][1] if len(a) > 0 else 0
                                        for a in sim_res.assignments]))

            results[video_id] = Result(query_id, video_id,
                                       status, actual_similar,
                                       float(sim_val), sim_res)
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


class Metrics:
    def __init__(self, results: Dict[int, Result]):
        self.results = results
        self._results_np = np.array([[r.actual, r.predicted]
                                     for r in results.values()])

    def conf_matrix(self):
        actual, predicted = self._results_np.T
        return sklearn.metrics.confusion_matrix(actual, predicted)

    def report(self):
        actual, predicted = self._results_np.T
        return sklearn.metrics.classification_report(actual, predicted)


async def prepare_query(query_id: int):
    async with CCWebVideoDataset() as ds:
        await ds.download_videos(ds.videos_from_query(query_id), os.path.join(ds.root_dir, "videos"))
        return ds


def conf_matrix_plot(conf_matrix):
    df = pd.DataFrame({"actual": ["false", "false", "true", "true"],
                       "predicted": ["false", "true", "false", "true"],
                       "entries": np.hstack(conf_matrix)})
    text_color = np.array(["black"] * len(df))
    text_color[df["entries"] < np.sum(df["entries"]) / 2] = "white"
    return (ggplot(df, aes(x="actual", y="predicted", fill="entries")) +
            geom_tile() +
            geom_text(aes(label="entries"), size=32, color=text_color) +
            labs(title="Confusion Matrix",
                 x="Actual",
                 y="Predicted"))


def calculate_roc_curve(results):
    results_np = np.array([[r.actual, r.similarity]
                           for r in results.values()])

    fpr, tpr, thres = sklearn.metrics.roc_curve(results_np[:,0], results_np[:, 1])
    roc = pd.DataFrame({"threshold": thres,
                        "fpr": fpr,
                        "tpr": tpr})
    return roc


def roc_curve_plot(measures):
    return (ggplot(measures, aes(x="fpr", y="tpr")) +
            geom_line(size=2, alpha=0.7, color="orange") +
            geom_point(alpha=0.35) +
            labs(title="ROC Curve",
                 x="False Positive Rate",
                 y="True Positive Rate"))


def workbook():
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
        roc = calculate_roc_curve(results)
        roc_curve_plot(roc).draw().show()
        pass
    pass


if __name__ == "__main__":
    workbook()
