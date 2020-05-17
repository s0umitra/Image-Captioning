import os
from pickle import dump

from src.lib.descrip_lib import load_clean_descriptions
from src.lib.libic import pick_load


def get_all_set(directory_path):
    dataset_all = os.listdir(directory_path)
    all_set = list()

    for line in dataset_all:
        # skip empty lines
        if len(line) < 1:
            continue

        # image identifier i.e. image name without extension
        i_name = line.split('.')[0]
        all_set.append(i_name)

    return set(all_set)


def minimize_words_count(captions):
    word_threshold = 10
    word_counts = dict()
    words_used = 0

    for word in captions:
        words_used += 1
        for w in word.split():
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_threshold]
    print('Minimized Vocabulary (Words) : %d -> %d' % (len(word_counts) + 1, len(vocab) + 1))

    int_to_word_mappings = dict()
    word_to_int_mappings = dict()

    integer = 1
    for w in vocab:
        word_to_int_mappings[w] = integer
        int_to_word_mappings[integer] = w
        integer += 1

    vocab_size = len(int_to_word_mappings) + 1
    data = vocab_size, word_to_int_mappings, int_to_word_mappings

    save_path = 'src\\mappings\\' + 'token_mappings.tk'
    dump(data, open(save_path, 'wb'))


def load_mappings():
    save_path = 'src\\mappings\\' + 'token_mappings.tk'

    while True:

        if os.path.exists(save_path):

            print('Old Word to Vector embeddings found, '
                  'Loading them!')
            return pick_load(save_path)

        else:

            path_dataset = "dataset\\flicker8k-dataset\\Flickr8k_Dataset\\Flicker8k_Dataset\\"
            path_desc = "src\\descriptions-generator\\output\\descriptions.txt"

            print('No Old Word to Vector embeddings found, '
                  'Creating a new one!')

            all_set = get_all_set(path_dataset)

            all_descriptions = load_clean_descriptions(path_desc, all_set)

            all_captions = []

            for key, val in all_descriptions.items():
                for cap in val:
                    all_captions.append(cap)

            minimize_words_count(all_captions)
