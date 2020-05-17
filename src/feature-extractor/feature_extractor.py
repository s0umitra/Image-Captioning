import os
from pickle import dump
from time import time
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

from src.lib.libic import init, set_opener
from src.lib.model_lib import feature_extractor


def initialize():

    # get program name
    caller = os.path.basename(__file__).split('.')[0]

    # initiate
    paths = init(caller)

    # set home path
    path_home = paths[0]
    os.chdir(path_home)

    # set paths
    path_dataset, \
        path_train_set, \
        path_test_set, \
        path_extracted_train_features, \
        path_extracted_test_features = paths[1]

    # Load and create a new model, by removing the last layer (output layer) from the inception v3
    model = InceptionV3(weights='imagenet')
    model_popped = Model(inputs=model.input, outputs=model.layers[-2].output)

    train_images = set_opener(path_train_set)

    test_images = set_opener(path_test_set)

    all_sets = [train_images, test_images]
    outputs = [path_extracted_train_features, path_extracted_test_features]

    ret = all_sets, path_dataset, model_popped, outputs

    return ret


def process_image(params):
    all_sets, path_dataset, model_popped, outputs = params

    total_count = 0

    # set initial time
    start_time = time()

    for i, dataset in enumerate(all_sets):
        count = 0
        features_encoded = dict()

        for name in dataset:
            count += 1

            name = name.strip()
            image_path = path_dataset + name

            feature_vector = feature_extractor(image_path, model_popped)

            image_name = name.split('.')[0]
            features_encoded[image_name] = feature_vector

            print('> Processing {}/{}'.format(count, len(dataset)) + ' : %s' % name)

        total_count += count

        # store to file
        dump(features_encoded, open(outputs[i], 'wb'))
        print("\nFeatures extracted :", len(features_encoded))
        print('Features saved to  :', outputs[i], end='\n\n')

    return total_count, start_time


if __name__ == '__main__':

    parameters = initialize()
    total, start = process_image(parameters)

    print("Total Features Extracted :", total)
    print("Processing Time          :", time() - start, "sec")
