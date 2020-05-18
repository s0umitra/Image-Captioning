import os
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model

from src.lib.cap_gen_lib import draw
from src.lib.libic import init
from src.lib.mapping_lib import load_mappings
from src.lib.model_lib import pred_caption_greedy, least_loss, feature_extractor


def initializer():
    # get program name
    caller = os.path.basename(__file__).split('.')[0]

    # initiate
    paths = init(caller)

    # set home path
    path_home = paths[0]
    os.chdir(path_home)


if __name__ == '__main__':

    initializer()

    model = InceptionV3(weights='imagenet')
    model_popped = Model(inputs=model.input, outputs=model.layers[-2].output)

    max_length = 34

    m_path, tot = least_loss()
    print("Total Models present  :", tot)
    print('Model with least loss :', m_path)
    print("Loading Model :", m_path.split('\\')[-1])

    use_model = load_model(m_path)

    vocab_size, word_to_int, int_to_word = load_mappings()

    for i in os.listdir('inputs'):

        image_name = 'inputs\\' + i

        img = feature_extractor(image_name, model_popped)

        pred_caption = pred_caption_greedy(img, use_model, max_length, word_to_int, int_to_word)
        draw(image_name, pred_caption)

        print("\nInput   :", i)
        print("Caption :", pred_caption)

    print("\nCheck output directory!")
