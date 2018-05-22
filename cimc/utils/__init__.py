import os
from typing import Union

import numpy as np
import torch
from PIL import Image

ImageType = Union[str, np.ndarray, Image.Image]

best_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_image(image: ImageType):
    if isinstance(image, Image.Image):
        img = image.convert('RGB')
    elif isinstance(image, str):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image, 'RGB')
    else:
        raise TypeError(f"image must be of type {ImageType}")
    return img


def simple_download(url: str, out: str = None, dir: str = None, force: bool = False):
    if out is None or len(out) < 1:
        out = os.path.basename(url)
    if dir is None or len(dir) < 1:
        dir = os.getcwd()
    elif not os.path.isabs(dir):
        dir = os.path.abspath(dir)
    assert os.path.isdir(dir), 'Invalid download directory'
    path = os.path.join(dir, out)

    if force or not os.access(path, os.W_OK):
        ret = os.system(f'wget -O "{path}" "{url}" >/dev/null 2>&1')
        if ret == 0:
            return path
        ret = os.system(f'curl "{url}" -o "{path}" >/dev/null 2>&1')
        if ret == 0:
            return path
        raise EnvironmentError('System has no available downloader')
    return path
