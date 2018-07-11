import os
import time

import torch
from torchvision.transforms import transforms as tf

import cimc.utils as utils
from cimc import resources
from cimc.models.yolov3_2.models import Darknet
from cimc.models.yolov3_2.utils.utils import non_max_suppression
from cimc.utils import bench

YOLOV3_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
YOLOV3_CFG = os.path.join(os.path.dirname(__file__), "yolov3.cfg")

_bench = bench.Bench("yolov3")


class YoloV3(Darknet):
    def __init__(self):
        super().__init__(YOLOV3_CFG)
        self.pre_process = tf.Compose([
            tf.ToTensor()
        ])

    def detect(self, image: utils.ImageType, confidence=0.25, nms_thres=0.4):
        m = _bench.measurements()
        t0 = time.time()

        img = utils.ToPILImage()(image)

        t0_1 = time.time()

        img = utils.SIMDResize((self.img_size, self.img_size))(img)

        t0_2 = time.time()

        img_input = self.pre_process(img).unsqueeze(0)

        t1 = time.time()

        device = next(self.parameters()).device
        img_input = img_input.to(device)

        t2 = time.time()

        with torch.no_grad():
            detections = self(img_input)

            t3 = time.time()

            detections = non_max_suppression(detections, 80, confidence, nms_thres)

            t4 = time.time()

            timings = {
                "to_image": t0_1 - t0,
                "resize": t0_2 - t0_1,
                "pre_process": t1 - t0_2,
                "gpu_transfer": t2 - t1,
                "predict": t3 - t2,
                "nms": t4 - t3,
                "total": t4 - t0,
            }
            (m
             .add("to.image", timings["to_image"])
             .add("resize", timings["resize"])
             .add("pre.process", timings["pre_process"])
             .add("gpu.transfer", timings["gpu_transfer"])
             .add("region.proposal", timings["predict"])
             .add("nms", timings["nms"])
             .add("iteration", timings["total"])).done()
            return detections, timings

    @classmethod
    def pre_trained(cls, weights_file: str = None):
        if weights_file is None:
            weights_file = resources.weight("yolov3.weights")
        utils.downloader.download_sync(YOLOV3_WEIGHTS_URL, weights_file)
        net = cls()
        net.load_weights(weights_file)
        return net


if __name__ == '__main__':
    import torch.onnx
    import numpy as np

    pt_net = YoloV3.pre_trained()
    x = torch.rand(1, 3, 416, 416, requires_grad=True)

    pt_out = torch.onnx._export(pt_net, x, "yolov3.onnx", export_params=True, verbose=True)

    ##########

    import onnx

    onnx_model = onnx.load("yolov3.onnx")

    import onnx_caffe2.backend

    c2_net = onnx_caffe2.backend.prepare(onnx_model)

    W = {onnx_model.graph.input[0].name: x.data.numpy()}

    c2_out = c2_net.run(W)[0]

    np.testing.assert_almost_equal(pt_out.detach().numpy(), c2_out, decimal=3)

    print(f"{x.shape}: torch={pt_out.shape}{pt_out[:5]}, c2={c2_out.shape}{c2_out[:5]}")
