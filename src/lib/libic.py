import os
from pickle import load
from sys import _getframe
import tensorflow as tf

from src.lib.set_paths import run_path_check


def init(caller):

    if caller in ['feature_extractor', 'model_trainer', 'model_evaluator', 'Caption_Generator']:

        if not caller == 'Caption_Generator':
            os.chdir('..//..')

        # setting GPU memory growth for no memory glitches
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    elif caller in ['generate_descriptions']:
        os.chdir('..//..')

    use_paths = run_path_check(caller)

    return use_paths


def get_program_name():

    return _getframe(1).f_code.co_filename.split('/')[-1].split('.')[0]


def set_opener(path):

    load_set = open(path, 'r')
    data = load_set.readlines()
    load_set.close()

    return data


def desc_loader(filename):

    load_desc = open(filename, 'r')
    data = load_desc.read()
    load_desc.close()

    return data


def pick_load(path):

    file = open(path, "rb")
    data = load(file)
    file.close()

    return data


def caption_creator(descriptions):

    captions = []
    for key, val in descriptions.items():
        for cap in val:
            captions.append(cap)

    return captions
