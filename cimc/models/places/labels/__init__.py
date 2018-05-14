import os

import numpy as np

FILE_DIR = os.path.dirname(__file__)


def _download(url: str, path: str = None, force: bool = False):
    path = path or os.path.join(FILE_DIR, os.path.basename(url))
    if force or not os.access(path, os.W_OK):
        os.system(f'wget -O "{path}" "{url}"')
    return path


def load_labels():
    # prepare all the labels
    # scene category relevant
    categories = _download(
        'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt')
    classes = list()
    with open(categories) as f:
        for line in f:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    io_places = _download('https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt')
    with open(io_places) as f:
        lines = f.readlines()
        io_labels = []
        for line in lines:
            items = line.rstrip().split()
            io_labels.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
    io_labels = np.array(io_labels)

    # scene attribute relevant
    attributes_places = _download(
        'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt')
    with open(attributes_places) as f:
        lines = f.readlines()
        attr_labels = [item.rstrip() for item in lines]

    attr_weights = _download('http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy')
    attr_weights = np.load(attr_weights)

    return classes, io_labels, attr_labels, attr_weights
