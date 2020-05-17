import os

from src.lib.descrip_lib import load_set, load_clean_descriptions, get_max_length
from src.lib.embedding_loader import emb_load
from src.lib.libic import get_program_name, init, caption_creator, pick_load
from src.lib.mapping_lib import load_mappings
from src.lib.model_lib import create_model, train_model, least_loss


def initializer():
    # get program name
    caller = get_program_name()

    # initiate
    paths = init(caller)

    # set home path
    path_home = paths[0]
    os.chdir(path_home)

    # set paths
    path_train_set, path_desc, path_extracted_train_features = paths[1]

    train_features = pick_load(path_extracted_train_features)
    train = load_set(path_train_set)

    train_descriptions = load_clean_descriptions(path_desc, train)
    print('Train Samples     : %d' % len(train_descriptions))

    all_train_captions = caption_creator(train_descriptions)
    print('Total Captions    :', len(all_train_captions))

    max_length = get_max_length(train_descriptions)
    print('Description Length: %d' % max_length)

    vocab_size, word_to_int, int_to_word = load_mappings()

    print('Loading Glove Word2Vec model, please wait...')
    embedding_dim, embedding_matrix = emb_load(vocab_size, word_to_int)

    return vocab_size, embedding_dim, embedding_matrix, max_length, train_descriptions, train_features, word_to_int


def model_training(params):

    vocab_size,\
        embedding_dim,\
        embedding_matrix,\
        max_length,\
        train_descriptions,\
        train_features,\
        word_to_int = params

    model = create_model(vocab_size, embedding_dim, embedding_matrix, max_length)

    # model = load_model('src\\model-trainer\\models\\model_1_9_(loss_2.134).h5')

    epochs = 20
    number_pics_per_bath = 3
    steps = len(train_descriptions)

    model_parameters_alpha = [train_descriptions, train_features, word_to_int, max_length, vocab_size]
    model_parameters_omega = [number_pics_per_bath, steps]

    train_model(1, model, epochs, model_parameters_alpha, model_parameters_omega)

    epochs = 10
    number_pics_per_bath = 6
    learning_rate = 0.0001

    extras = [learning_rate]
    model_parameters_omega = [number_pics_per_bath, steps, extras]

    train_model(2, model, epochs, model_parameters_alpha, model_parameters_omega)

    epochs = 5
    number_pics_per_bath = 12
    learning_rate = 0.0001

    extras = [learning_rate]
    model_parameters_omega = [number_pics_per_bath, steps, extras]

    train_model(3, model, epochs, model_parameters_alpha, model_parameters_omega)


if __name__ == '__main__':

    parameters = initializer()
    model_training(parameters)
    min_model, all_models = least_loss()

    print("Total Models present  :", all_models)
    print('Model with least loss :', min_model)
