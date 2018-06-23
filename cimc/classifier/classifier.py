import logging
import os
import pickle
import time
from typing import List, Optional

import imageio
from imageio.core import Format
from torchvision.transforms import transforms as tf
from tqdm import tqdm

from cimc import resources, utils
from cimc.classifier.annotator import annotate_video
from cimc.classifier.utils import get_clsf
from cimc.models import YoloV3_2 as YoloV3
from cimc.models.yolov3.labels import COCO_LABELS
from cimc.scene import SceneDetector
from cimc.scene.classification import SceneClassifier
from cimc.tracker import MultiTracker
from cimc.utils import bbox, log
from .classification import VideoClassification, Segment

logger = logging.getLogger(__name__)
logger.addHandler(log.TqdmLoggingHandler())
logger.setLevel(logging.DEBUG)


def classify_video(video_uri: str) -> VideoClassification:
    logger.info(f"Starting classification for video '{video_uri}'")

    with imageio.get_reader(video_uri, "ffmpeg") as video:  # type: Format.Reader
        meta = video.get_meta_data()
        size = meta["size"]
        fps = meta["fps"]
        length = len(video)
        duration = utils.duration_str(length / fps)

        logger.info(
            f"'{os.path.basename(video_uri)}' : "
            f"{size[0]}x{size[1]} {duration}({fps}fps {length} frames)"
        )

        detections_uri = f"{video_uri}.dets"
        detections: List[List[bbox.BoundingBox]] = []
        gen_detections = True
        if os.path.isfile(detections_uri):
            logger.debug(f"Found detections file {detections_uri}")
            try:
                with open(detections_uri, "rb") as dets_fd:
                    detections = pickle.load(dets_fd)
                gen_detections = False
            except pickle.UnpicklingError as e:
                logger.warning(
                    f"The detections file {detections_uri} "
                    f"is invalid. YOLOv3 will be used"
                )

        scene_detector = SceneDetector(downscale=4, min_length=int(fps))

        scene_classifier = SceneClassifier(step=int(fps / 2))

        yolov3_net: Optional[YoloV3] = None
        if gen_detections:
            yolov3_net = YoloV3.pre_trained().to(utils.best_device)
            pp = tf.Compose(
                [bbox.ReverseScale(*size), bbox.FromYoloOutput(COCO_LABELS)]
            )

        tracker = MultiTracker(max_age=int(fps), min_hits=int(fps / 2), iou_thres=0.35)

        clsf = VideoClassification(video_uri, metadata=meta)
        t_start = time.time()
        with tqdm(
            total=length,
            desc=f"Classifying '{video_uri}'",
            dynamic_ncols=True,
            unit="frame",
        ) as bar:
            segment: Optional[Segment] = None
            i = 0
            while i < length:
                try:
                    frame = video.get_next_data()
                except imageio.core.CannotReadFrameError:
                    break
                bar.update()
                if scene_detector.update(frame):
                    if segment is not None:
                        segment.end = i
                        segment.scene = scene_classifier.classification()
                        clsf.append_segment(segment)
                    segment = Segment(id=len(clsf.segments), start=i)
                    scene_classifier.reset()
                    tracker.reset()

                scene_classifier.update(frame)

                if gen_detections:
                    boxes = yolov3_net.detect(frame)[0][0]
                    # print(len(boxes))
                    # print(boxes[0])
                    # print("------ CONVERTING TO BoundingBox ------")
                    bboxes = [pp(box) for box in boxes]
                    # print(bboxes[0])
                    detections.append(bboxes)
                else:
                    bboxes = detections[i]
                objects = tracker.update(bboxes)
                segment.append_objects(objects)
                i += 1

            segment.end = length
            segment.scene = scene_classifier.classification()
            clsf.append_segment(segment)

        t_end = time.time()

        if gen_detections:
            with open(detections_uri, "wb") as dets_fd:
                pickle.dump(detections, dets_fd)
            logger.debug(f"Saved YOLOv3 detections as '{detections_uri}'")

        time_taken = utils.duration_str(t_end - t_start)
        logger.info(
            f"Finished classifying '{video_uri}' in {time_taken} (found {len(clsf.segments)} scenes)"
        )

        return clsf


def classify_and_annotate(video_uri=None, clsf_uri=None):
    if video_uri is None:
        video_uri = resources.video("goldeneye.mp4")

    clsf = get_clsf(clsf_uri, video_uri)
    annotate_video(video_uri, clsf)


if __name__ == "__main__":
    # try:
    #     with open(resources.video('justice-league.mp4.clsf'), 'rb') as fd:
    #         clsf = pickle.load(fd)
    #         pass
    # except:
    #     pass
    # get_clsf(video_uri=resources.video('goldeneye-justiceleague.mp4'))
    # classify_and_annotate(resources.video("Venice-1.bk.mp4"))
    # classify_and_annotate(resources.video('goldeneye-justiceleague.mp4'))
    # classify_and_annotate(resources.video('goldeneye.mp4'))
    # classify_and_annotate(resoUrces.video('goldeneye-2x.mp4'))
    # classify_and_annotate(resources.video('TUD-Campus.mp4'))
    classify_and_annotate(resources.video("TUD-Campus.mp4"))
    # classify_and_annotate(resources.video('TUD-Campus.var.rotate-scale.mp4'))
    # classify_and_annotate(resources.video('TUD-Crossing.mp4'))
    # classify_and_annotate(resources.video('ADL-Rundle-8.mp4'))
    # classify_and_annotate(resources.video('Venice-1.mp4'))
    # classify_and_annotate(resources.video('justice-league.mp4'))
    # classify_and_annotate(resources.video('deadpool2.mp4'))
    # classify_and_annotate(resources.video('ant-man-and-wasp.mp4'))
    # classify_and_annotate(resources.video('bvs.mp4'))
    # classify_and_annotate(resources.video('goldeneye.mp4'))
    # classify_and_annotate(resources.video(
    #     'TUD-Campus.var.rotate-scale-flip-color.mp4'))
