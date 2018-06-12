import logging
import os.path
import pickle
from typing import Optional

from cimc.classifier import VideoClassification
from cimc.utils import log

logger = logging.getLogger(__name__)
logger.addHandler(log.TqdmLoggingHandler())
logger.setLevel(logging.DEBUG)


def load_clsf(clsf_uri: str) -> Optional[VideoClassification]:
    try:
        with open(clsf_uri, "rb") as fd:
            return pickle.load(fd)
    except pickle.PickleError as e:
        return None


def get_clsf(clsf_uri: str = None, video_uri: str = None) -> VideoClassification:
    clsf: VideoClassification = None
    if clsf_uri is not None:
        if not os.path.isfile(clsf_uri):
            logger.warning(f"Provided clsf uri doesn't exist: {clsf_uri}")
        else:
            clsf = load_clsf(clsf_uri)
            if clsf is not None:
                logger.info(f"Loaded clsf: {clsf_uri}")
            else:
                logger.warning(f"Invalid clsf provided: {clsf_uri}")
    if clsf is None:
        clsf_uri = f"{video_uri}.clsf"
        if os.path.isfile(clsf_uri):
            clsf = load_clsf(clsf_uri)
            if clsf is not None:
                logger.info(f"Found and loaded existing clsf: {clsf_uri}")
        if clsf is None:
            if video_uri is None or not os.path.isfile(video_uri):
                raise FileNotFoundError(video_uri)
            from .classifier import classify_video
            clsf = classify_video(video_uri)
            with open(clsf_uri, "wb") as fd:
                pickle.dump(clsf, fd)
                logger.info(f"Saved classification to '{clsf_uri}'")
    return clsf
