import os
from pickle import load
from keras.engine.saving import load_model
from nltk.translate.bleu_score import corpus_bleu

from src.lib.descrip_lib import load_clean_descriptions, load_set, get_max_length
from src.lib.libic import get_program_name, init
from src.lib.mapping_lib import get_all_set, load_mappings
from src.lib.model_lib import pred_caption_greedy, least_loss


def initializer():
    # get program name
    caller = get_program_name()

    # initiate
    paths = init(caller)

    # set home path
    path_home = paths[0]
    os.chdir(path_home)

    # set paths
    path_dataset, \
        path_test_set, \
        path_desc, \
        path_extracted_test_features = paths[1]

    test_features = load(open(path_extracted_test_features, "rb"))
    test = load_set(path_test_set)

    test_descriptions = load_clean_descriptions(path_desc, test)
    print('Test Samples      : %d' % len(test_descriptions))

    all_set = get_all_set(path_dataset)
    all_descriptions = load_clean_descriptions(path_desc, all_set)
    print('Total Samples     : %d' % len(all_descriptions))

    # determine the maximum sequence length
    max_length = get_max_length(all_descriptions)
    print('Description Length: %d' % max_length)

    vocab_size, word_to_int, int_to_word = load_mappings()

    return test_descriptions, test_features, max_length, word_to_int, int_to_word


def load_least_model():
    m_path, tot = least_loss()
    print("Total Models present  :", tot)
    print('Model with least loss :', m_path)

    print("Evaluating Model :", m_path.split('\\')[-1])
    least_loss_model = load_model(m_path)

    return least_loss_model


def evaluate_model(eval_model, eval_params):

    descriptions, features, max_length, word_to_int, int_to_word = eval_params
    actual, predicted = list(), list()
    
    count = 0

    for key, desc_list in descriptions.items():
        # generate description
        count += 1
        print('Eval Progress : {}/{}'.format(count, len(descriptions)))

        y_hat = pred_caption_greedy(features[key], eval_model, max_length, word_to_int, int_to_word)

        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(y_hat.split())

    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


if __name__ == '__main__':

    params = initializer()
    model = load_least_model()
    evaluate_model(model, params)

    print("-------DONE-------")
