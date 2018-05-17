import functools as f
import time
from enum import Enum
from typing import List, Dict

import attr
import numpy as np
import torch
from scipy.misc import imresize as imresize
from torch.nn import functional as F
from torchvision import transforms as tf

import cimc.resources as resources
import cimc.utils as utils
from .labels import load_labels
from .wideresnet import ResNet, BasicBlock


class SceneType(int, Enum):
    INDOOR = 0
    OUTDOOR = 1


@attr.s(slots=True)
class CategoryPrediction:
    name: str = attr.ib()
    confidence: float = attr.ib()


@attr.s(slots=True, str=False)
class SceneClassification:
    type: SceneType = attr.ib()
    categories: List[CategoryPrediction] = attr.ib(factory=list)
    attributes: List[str] = attr.ib(factory=list)
    timings: Dict[str, float] = attr.ib(factory=dict)

    def __str__(self):
        tmp = f"┌─SCENE CLASSIFICATION\n" \
              f"├─TYPE: {self.type.name}\n" \
              f"{'├' if len(self.attributes) > 0 else '└'}─CATEGORIES:\n"
        for cat in self.categories:
            tmp += f"│   {cat.name}({cat.confidence*100:.2f}%)\n"
        if len(self.attributes) > 0:
            tmp += f"└─ATTRIBUTES:\n" \
                   f"    {', '.join(sorted(self.attributes))}"
        return tmp


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


class Places365(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=365)
        self.pre_process = tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        classes, io_labels, attr_labels, attr_weights = load_labels()
        self.classes = classes
        self.io_labels = io_labels
        self.attr_labels = attr_labels
        self.attr_weights = attr_weights

    def classify(self, image: utils.ImageType) -> SceneClassification:
        img = utils.to_image(image)

        if self.training:
            self.eval()

        features = {}
        hooks = {}

        def hook(name, _m, _i, o):
            features[name] = np.squeeze(o.detach().cpu().numpy())

        hooks['layer4'] = self.layer4.register_forward_hook(f.partial(hook, 'layer4'))
        hooks['avgpool'] = self.avgpool.register_forward_hook(f.partial(hook, 'avgpool'))

        # get the softmax weight
        params = list(self.parameters())
        weight_softmax = params[-2].data.cpu().numpy()
        weight_softmax[weight_softmax < 0] = 0
        t1 = time.time()
        img_input = self.pre_process(img).unsqueeze(0).to(next(self.parameters()).device)

        with torch.no_grad():
            t2 = time.time()
            logit = self(img_input)
            t3 = time.time()

            h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            probs, idx = probs.cpu().numpy(), idx.cpu().numpy()

            io_image = np.mean(self.io_labels[idx[:10]])  # vote for the indoor or outdoor
            responses_attribute = self.attr_weights.dot(features['avgpool'])
            idx_a = np.argsort(responses_attribute)
            t4 = time.time()

            env_type = SceneType.INDOOR if io_image < 0.5 else SceneType.OUTDOOR
            cats = [CategoryPrediction(cls, prob) for prob, cls in zip(probs[:5], self.classes[idx[:5]])]
            attrs = list(self.attr_labels[idx_a[-1:-10:-1]])
            t5 = time.time()
            timings = {
                'pre_process': t2 - t1,
                'forward_pass': t3 - t2,
                'result_prep': t4 - t3,
                'result_creation': t5 - t4,
                'total': t5 - t1
            }
            result = SceneClassification(env_type, cats, attrs, timings)
            for h in hooks.values():
                h.remove()
            return result

    @classmethod
    def pre_trained(cls, model_file: str = None):
        if model_file is None:
            model_file = resources.weight("widerestnet18_places365.pth.tar")
        utils.simple_download('http://places2.csail.mit.edu/models_places365/wideresnet18_places365.pth.tar',
                              model_file)
        m = cls()
        checkpoint = torch.load(model_file)
        m.load_state_dict({str.replace(k, 'module.', ''): v
                           for k, v in checkpoint['state_dict'].items()})
        return m


def main():
    model = Places365.pre_trained()
    model.to(utils.best_device)
    n_timings = 50
    timings = np.empty(n_timings)
    test_image = resources.image('places-test.jpg')
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
    # generate class activation mapping (CAM)
    # print('Class activation map is saved as cam.jpg')
    # CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    # img = cv2.imread('test.jpg')
    # height, width, _ = img.shape
    # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    # result = heatmap * 0.4 + img * 0.5
    # cv2.imwrite('cam.jpg', result)


if __name__ == '__main__':
    main()
