import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize as imresize
from torch.nn import functional as F
from torchvision import transforms as tf

from .labels import load_labels


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


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    from . import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    model.eval()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(_module, _input, output):
        features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    features_names = ['layer4', 'avgpool']  # this is the last conv layer of the resnet
    for name in features_names:
        # noinspection PyProtectedMember
        model._modules.get(name).register_forward_hook(hook_feature)
    return model, features_blobs


def main():
    # load the labels
    classes, io_labels, attr_labels, attr_weights = load_labels()

    # load the model
    model, features_blobs = load_model()

    # load the transformer
    pre_process = tf.Compose([
        tf.Resize((224, 224)),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax < 0] = 0

    # load the test image
    img_url = 'http://places.csail.mit.edu/demo/6.jpg'
    os.system(f'wget {img_url} -q -O test.jpg')
    img = Image.open('test.jpg')
    input_img = pre_process(img).unsqueeze(0)

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    print('RESULT ON ' + img_url)

    # output the IO prediction
    io_image = np.mean(io_labels[idx[:10]])  # vote for the indoor or outdoor
    env_type = 'indoor' if io_image < 0.5 else 'outdoor'
    print(f"-TYPE OF ENVIRONMENT: {env_type}")

    # output the prediction of scene category
    print('--SCENE CATEGORIES:')
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # output the scene attributes
    responses_attribute = attr_weights.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    print('--SCENE ATTRIBUTES:')
    print(', '.join([attr_labels[idx_a[i]] for i in range(-1, -10, -1)]))

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
