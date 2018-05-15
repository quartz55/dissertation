from typing import Union

import numpy as np
from PIL import Image

ImageType = Union[str, np.ndarray, Image.Image]


def to_image(image: ImageType):
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, str):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image, 'RGB')
    else:
        raise TypeError(f"image must be of type {ImageType}")
    return img
