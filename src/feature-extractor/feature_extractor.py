import os
import numpy as np
from pickle import dump, load
import sys
from keras.preprocessing import image
from time import time
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
import tensorflow as tf

from src.lib.path_check import run_path_check


# setting GPU memory growth for no memory glitches
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Get and verify Paths
caller = os.path.basename(__file__).split('.')[0]
use_paths, status = run_path_check(caller)
if not status:
    sys.exit()
path_dataset = use_paths[0]
path_train_set = use_paths[1]
path_extracted_features = 'output\\extracted_train_features.ed'

# Load and create a new model, by removing the last layer (output layer) from the inception v3
model = InceptionV3(weights='imagenet')
model_popped = Model(inputs=model.input, outputs=model.layers[-2].output)

features_encoded = dict()

train_set = open(path_train_set)
train_images = train_set.readlines()

# to get total time
start_time = time()

for name in train_images:

    name = name.strip()
    image_path = path_dataset + name

    print('> Processing : %s' % name)

    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    feature_vector = model_popped.predict(x)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])

    image_name = name.split('.')[0]
    features_encoded[image_name] = feature_vector

print("Total Files processed : ", len(train_images))
print("Total Time required   : ", time()-start_time)

# store to file
dump(features_encoded, open(path_extracted_features, 'wb'))
