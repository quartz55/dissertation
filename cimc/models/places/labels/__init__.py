import os

import numpy as np

from cimc.utils import simple_download

FILE_DIR = os.path.dirname(__file__)
CATEGORIES_URL = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
IO_URL = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
ATTRIBUTES_URL = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
ATTRS_WEIGHTS_URL = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'

category_type = np.dtype([('id', np.int32),
                          ('label', np.unicode, 40),
                          ('type', np.uint8)])
attribute_type = np.dtype([('id', np.int32),
                           ('label', np.unicode, 40),
                           ('weights', np.float64, (512,))])

CATEGORIES: np.ndarray
ATTRIBUTES: np.ndarray


# 0 is indoor, 1 is outdoor
def _load_labels():
    global CATEGORIES
    categories = simple_download(CATEGORIES_URL, dir=FILE_DIR)
    io_places = simple_download(IO_URL, dir=FILE_DIR)
    with open(categories) as cats_f:
        with open(io_places) as io_f:
            cats, io = cats_f.readlines(), io_f.readlines()
            CATEGORIES = np.empty(len(cats), category_type)
            for i, cat in enumerate(zip(cats, io)):
                cat_label = cat[0].strip().split(' ')[0][3:]
                cat_type = int(cat[1].rstrip().split()[-1]) - 1
                CATEGORIES[i] = (i, cat_label, cat_type)

    attr_weights = simple_download(ATTRS_WEIGHTS_URL, dir=FILE_DIR)
    attr_weights = np.load(attr_weights)

    attributes_places = simple_download(ATTRIBUTES_URL, dir=FILE_DIR)
    with open(attributes_places) as f:
        lines = f.readlines()
        attr_labels = np.array([item.rstrip() for item in lines])

    global ATTRIBUTES
    ATTRIBUTES = np.empty(len(attr_labels), attribute_type)
    ATTRIBUTES['id'] = range(len(attr_labels))
    ATTRIBUTES['label'] = attr_labels
    ATTRIBUTES['weights'] = attr_weights


_load_labels()

if __name__ == '__main__':
    pass
