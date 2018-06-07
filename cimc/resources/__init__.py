import os.path

from pkg_resources import resource_filename

from .annotations import load_similarities

__BASE_PKG = 'cimc.resources'


def video(name: str = '') -> str:
    return __get_resource_filepath('videos', name)


def image(name: str = '') -> str:
    return __get_resource_filepath('images', name)


def font(name: str = '') -> str:
    return __get_resource_filepath('fonts', name)


def weight(name: str = '') -> str:
    return __get_resource_filepath('weights', name)


def annotation(name: str = '') -> str:
    return __get_resource_filepath('annotations', name)


def exists(path: str) -> bool:
    return os.path.exists(path)


def __get_resource_filepath(module: str, file_name: str) -> str:
    module = f'{__BASE_PKG}.{module}'
    module_path = resource_filename(module, '')
    return os.path.join(module_path, file_name)
