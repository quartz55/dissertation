import os

import numpy as np

from cimc.utils import simple_download

FILE_DIR = os.path.dirname(__file__)
CATEGORIES_URL = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
IO_URL = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
ATTRIBUTES_URL = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
ATTRS_WEIGHTS_URL = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'


def load_labels():
    categories = simple_download(CATEGORIES_URL, dir=FILE_DIR)
    classes = list()
    with open(categories) as f:
        for line in f:
            classes.append(line.strip().split(' ')[0][3:])
    classes = np.array(classes)

    io_places = simple_download(IO_URL, dir=FILE_DIR)
    with open(io_places) as f:
        lines = f.readlines()
        io_labels = []
        for line in lines:
            items = line.rstrip().split()
            io_labels.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
    io_labels = np.array(io_labels)

    attributes_places = simple_download(ATTRIBUTES_URL, dir=FILE_DIR)
    with open(attributes_places) as f:
        lines = f.readlines()
        attr_labels = np.array([item.rstrip() for item in lines])

    attr_weights = simple_download(ATTRS_WEIGHTS_URL, dir=FILE_DIR)
    attr_weights = np.load(attr_weights)

    return classes, io_labels, attr_labels, attr_weights
