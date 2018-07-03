import functools as f
import time
from enum import Enum
from typing import Dict

import attr
import numpy as np
import torch
from scipy.misc import imresize as imresize
from torch.nn import functional as F
from torchvision.transforms import transforms as tf

import cimc.resources as resources
import cimc.utils as utils
import cimc.utils.downloader as downloader
from cimc.utils import bench
from . import labels as lbl
from .wideresnet import WideResNet, BasicBlock


class SceneType(int, Enum):
    INDOOR = 0
    OUTDOOR = 1


category_pred_type = np.dtype(
    [("id", np.int32), ("label", np.unicode, 40), ("confidence", np.float32)]
)

attribute_pred_type = np.dtype([("id", np.int32), ("label", np.unicode, 40)])


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(imresize(cam_img, size_upsample))
    return output_cam


@attr.s(slots=True, str=False)
class PlacesClassification:
    type: SceneType = attr.ib()
    categories: np.ndarray = attr.ib(
        factory=lambda: np.empty(0, dtype=category_pred_type)
    )
    attributes: np.ndarray = attr.ib(
        factory=lambda: np.empty(0, dtype=attribute_pred_type)
    )
    timings: Dict[str, float] = attr.ib(factory=dict)

    def __str__(self):
        tmp = f"┌─PLACES CLASSIFICATION\n" f"├─TYPE: {self.type.name}\n" f"├─CATEGORIES:\n"
        for cat in self.categories:
            tmp += f"│   {cat['name']}({cat['confidence']*100:.2f}%)\n"
        if len(self.attributes) > 0:
            tmp += f"└─ATTRIBUTES:\n" f"    {', '.join(sorted(self.attributes['label']))}"
        return tmp


_bench = bench.Bench("places365")


class Places365(WideResNet):
    def __init__(self):
        # 18 layers
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=365)
        # 34 layers
        # super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=365)
        # 50 layers
        # super().__init__(Bottleneck, [3, 4, 6, 3], num_classes=365)
        # 101 layers
        # super().__init__(Bottleneck, [3, 4, 23, 3], num_classes=365)
        # 152 layers
        # super().__init__(Bottleneck, [3, 8, 36, 3], num_classes=365)
        self.pre_process = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        self._prepared = False
        self._features = {}
        self._hooks = {}

    def prepare(self):
        self.eval()

        def hook(name, _m, _i, o):
            self._features[name] = np.squeeze(o.detach().cpu().numpy())

        self._features = {}
        self._hooks = {
            "layer4": self.layer4.register_forward_hook(f.partial(hook, "layer4")),
            "avgpool": self.avgpool.register_forward_hook(f.partial(hook, "avgpool"))
        }

    def classify(self, image: utils.ImageType) -> PlacesClassification:
        t0 = time.time()

        img = utils.ToPILImage()(image)

        t0_1 = time.time()

        img = utils.SIMDResize((224, 244))(img)

        t0_2 = time.time()

        img_input = self.pre_process(img).unsqueeze(0)

        t1 = time.time()

        curr_device = next(map(lambda p: p.device, self.parameters()), utils.best_device)
        img_input = img_input.to(curr_device)

        t2 = time.time()

        with torch.no_grad():
            logit = self(img_input)

            t3 = time.time()

            h_x = F.softmax(logit, 1).data.squeeze()
            probs_cats, cats_idx = h_x.sort(0, True)
            probs_cats, cats_idx = probs_cats.cpu().numpy(), cats_idx.cpu().numpy()
            top_5_cats = lbl.CATEGORIES[cats_idx[:5]]

            io_image = np.mean(lbl.CATEGORIES[cats_idx[:10]]["type"])

            responses_attribute = lbl.ATTRIBUTES["weights"].dot(self._features["avgpool"])
            attrs_idx = np.argsort(responses_attribute)
            top_10_attrs = lbl.ATTRIBUTES[attrs_idx[-1:-10:-1]]

            t4 = time.time()

            env_type = SceneType.INDOOR if io_image < 0.5 else SceneType.OUTDOOR
            categories = np.fromiter(
                (
                    (id, label, conf)
                    for (id, label), conf in zip(
                    top_5_cats[["id", "label"]], probs_cats[:5]
                )
                ),
                dtype=category_pred_type,
            )
            attributes = top_10_attrs[["id", "label"]].astype(attribute_pred_type)

            t5 = time.time()
            timings = {
                "to.image": t0_1 - t0,
                "resize": t0_2 - t0_1,
                "pre.process": t1 - t0_2,
                "gpu.transfer": t2 - t1,
                "forward.pass": t3 - t2,
                "result.prep": t4 - t3,
                "result.creation": t5 - t4,
                "total": t5 - t0,
            }
            result = PlacesClassification(
                type=env_type,
                categories=categories,
                attributes=attributes,
                timings=timings,
            )

            m = _bench.measurements()
            for k, v in timings.items():
                m.add(k, v)
            m.done()

            return result

    @classmethod
    def pre_trained(cls, model_file: str = None):
        if model_file is None:
            model_file = resources.weight("wideresnet18_places365.pth.tar")
        downloader.download_sync(
            "http://places2.csail.mit.edu/models_places365/wideresnet18_places365.pth.tar",
            model_file,
        )
        m = cls()
        checkpoint = torch.load(model_file, map_location=lambda st, loc: st)
        m.load_state_dict(
            {
                str.replace(k, "module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        )
        return m


def main():
    model = Places365.pre_trained()
    model.to(utils.best_device)
    n_timings = 50
    timings = np.empty(n_timings)
    test_image = resources.image("places-test.jpg")
    # Warmup
    for i in range(5):
        model.classify(test_image)
    for i in range(n_timings):
        t1 = time.time()
        res = model.classify(test_image)
        timings[i] = time.time() - t1
    avg_time = np.mean(timings)
    print(f"{avg_time*1e3:.4f}ms")
    print(res)
    print(res.timings)


if __name__ == "__main__":
    main()
