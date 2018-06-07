import yaml
import glob
import os


def load_similarities():
    base_dir = 'similarities'
    base_dir = os.path.join(os.path.dirname(__file__), base_dir)
    filenames = glob.glob(f"{base_dir}/*.yaml")
    similarities = {}
    for path in filenames:
        with open(path, 'r') as file:
            name = os.path.splitext(os.path.basename(path))[0]
            similarities[name] = yaml.load(file)
    return similarities
