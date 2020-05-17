import os
import numpy as np

from src.lib.set_paths import run_path_check


def emb_load(vocab_size, word_to_int):

    caller = os.path.basename(__file__).split('.')[0]
    use_paths = run_path_check(caller)

    path_glove_txt = use_paths[1][0]

    embeddings_index = {}
    f = open(path_glove_txt, encoding="utf-8")

    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients

    f.close()

    print('Found %s word vectors' % len(embeddings_index))

    embedding_dim = 200

    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_to_int.items():
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_dim, embedding_matrix
