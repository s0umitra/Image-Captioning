import os
import numpy as np
from keras import Input, Model
from keras.backend import set_value
from keras.layers import Dropout, Embedding, Dense, LSTM, add
from keras.utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from keras_preprocessing.sequence import pad_sequences
from numpy import array


def create_model(vocab_size, embedding_dim, embedding_matrix, max_length):

    inputs_image = Input(shape=(2048,))
    feature_layer_1 = Dropout(0.2)(inputs_image)
    feature_layer_2 = Dense(256, activation='relu')(feature_layer_1)

    inputs_sequence = Input(shape=(max_length,))
    sequence_layer_1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs_sequence)
    sequence_layer_2 = Dropout(0.2)(sequence_layer_1)
    sequence_layer_3 = LSTM(256)(sequence_layer_2)

    decoder1 = add([feature_layer_2, sequence_layer_3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs_image, inputs_sequence], outputs=outputs)

    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def data_generator(descriptions, image, word_to_int, max_length, num_photos_per_batch, vocab_size):

    list_photos = list()
    list_in_seq = list()
    list_out_seq = list()
    n = 0

    while True:

        for key, desc_list in descriptions.items():
            n += 1

            photo = image[key]

            for desc in desc_list:

                seq = [word_to_int[word] for word in desc.split(' ') if word in word_to_int]

                for i in range(1, len(seq)):

                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    list_photos.append(photo)
                    list_in_seq.append(in_seq)
                    list_out_seq.append(out_seq)

            if n == num_photos_per_batch:

                yield [[array(list_photos), array(list_in_seq)], array(list_out_seq)]
                list_photos, list_in_seq, list_out_seq = list(), list(), list()
                n = 0


def train_model(idn, model, epochs, model_parameters_alpha, model_parameters_omega):
    path_model = 'src\\model-trainer\\models\\'

    train_descriptions = model_parameters_alpha[0]
    train_features = model_parameters_alpha[1]
    word_to_int = model_parameters_alpha[2]
    max_length = model_parameters_alpha[3]
    vocab_size = model_parameters_alpha[4]

    number_pics_per_bath = model_parameters_omega[0]
    steps = model_parameters_omega[1]

    if len(model_parameters_omega) == 3:
        extras = model_parameters_omega[2]
        set_value(model.optimizer.lr, extras[0])

    for i in range(epochs):
        generator = data_generator(train_descriptions,
                                   train_features,
                                   word_to_int,
                                   max_length,
                                   number_pics_per_bath,
                                   vocab_size
                                   )

        history = model.fit_generator(generator,
                                      epochs=1,
                                      steps_per_epoch=steps,
                                      verbose=1,
                                      )

        # pull out metrics from the model
        loss = history.history.get('loss')[0]

        # model naming
        model_name = 'model_' + str(idn) + '_' + str(i) + '_(loss_%.3f' % loss + ').h5'

        # saving the model to local storage
        model.save(path_model + str(model_name))
        print('\nModel saved : ' + model_name, end="\n\n")


def least_loss():
    path = 'src\\model-trainer\\models\\'

    all_models = os.listdir(path)
    all_models_loss = [x.split('(')[1].split(')')[0].split('_')[1] for x in all_models]
    least_loss_model = [x for x in all_models if x.split('(')[1].split(')')[0].split('_')[1] == min(all_models_loss)]

    return str(path + least_loss_model[0]), str(len(all_models))


def pred_caption_greedy(photo, model, max_length, word_to_int, int_to_word):

    photo = np.array(photo)
    photo = np.expand_dims(photo, axis=0)
    in_text = '<start>'

    for i in range(max_length):

        sequence = [word_to_int[w] for w in in_text.split() if w in word_to_int]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_hat = model.predict([photo, sequence], verbose=0)

        y_hat = np.argmax(y_hat)
        word = int_to_word[y_hat]
        in_text += ' ' + word

        if word == '<end>':
            break

    pred_caption = in_text.split()
    pred_caption = pred_caption[1:-1]
    pred_caption = ' '.join(pred_caption)

    return pred_caption


def feature_extractor(image, in_model):

    img = load_img(image, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    ext_ft = in_model.predict(x)
    ext_ft = np.reshape(ext_ft, ext_ft.shape[1])

    return ext_ft
