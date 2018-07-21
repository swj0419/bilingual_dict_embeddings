#!/usr/bin/env python3

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, CuDNNGRU, Bidirectional, \
    Merge, BatchNormalization, merge, Conv1D, Dot, Multiply, Lambda, Subtract, TimeDistributed
from keras.layers.core import Flatten, Reshape
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.layers import Input
import tensorflow as tf
from keras.initializers import RandomNormal
from keras.layers.merge import dot
from keras.optimizers import TFOptimizer
from keras.utils import plot_model


def get_model(
        nb_word,
        dim,
        length,
        desc_length,
        word_emb_matrix=None,
        context_emb_matrix=None,
        word_emb_trainable=True,
        context_emb_trainable=True,
):
    # parameter
    def make_emb(emb_matrix=None, trainable=True):
        if emb_matrix is not None:
            return Embedding(
                input_dim=nb_word,
                output_dim=dim,
                weights=[emb_matrix],
                trainable=trainable,
            )
        else:
            return Embedding(
                input_dim=nb_word,
                output_dim=dim,
                embeddings_initializer=RandomNormal(
                    mean=0.0,
                    stddev=1.0 / dim,
                    seed=None,
                ),
                trainable=trainable,
            )

    word_emb = make_emb(word_emb_matrix, word_emb_trainable)
    context_emb = make_emb(context_emb_matrix, context_emb_trainable)

    # word2vec part. note that two langauges are dealt
    # in the same word2vec func.
    word_input = Input(shape=(1, ))
    context_input = Input(shape=(1, ))

    word_embedded = word_emb(word_input)
    context_embedded = context_emb(context_input)

    output = Dot(axes=-1)([word_embedded, context_embedded])
    output = Flatten()(output)

    word2vec_model = Model(inputs=[word_input, context_input], outputs=output)

    word2vec_model_infer = Model(
        inputs=[word_input], outputs=Flatten()(word_embedded))

    # bilbowa
    sent_0_input = Input(shape=(length, ))
    mask_0_input = Input(shape=(length, ))
    sent_1_input = Input(shape=(length, ))
    mask_1_input = Input(shape=(length, ))

    sent_0_embedded = word_emb(sent_0_input)
    sent_1_embedded = word_emb(sent_1_input)

    def encode_function(x):
        sent_embedded, mask = x
        sent_embedded = sent_embedded * K.expand_dims(mask, -1)
        sent_encoded = K.sum(sent_embedded, axis=-2, keepdims=False)

        use_avg = True
        if use_avg:
            sent_encoded = sent_encoded / K.sum(mask, axis=-1, keepdims=True)
        return sent_encoded

    encode = Lambda(encode_function)

    sent_0_encoded = encode([sent_0_embedded, mask_0_input])
    sent_1_encoded = encode([sent_1_embedded, mask_1_input])

    diff_sent_encoded = Subtract()([sent_0_encoded, sent_1_encoded])

    def scale_diff_function(x):
        diff_sent_encoded, mask_0, mask_1 = x
        t = (K.sum(mask_0, axis=-1, keepdims=True) + K.sum(
            mask_1, axis=-1, keepdims=True)) * 0.5
        return diff_sent_encoded * t

    diff_sent_encoded = Lambda(scale_diff_function)([
        diff_sent_encoded,
        mask_0_input,
        mask_1_input,
    ])

    bilbowa_model = Model(
        inputs=[sent_0_input, mask_0_input, sent_1_input, mask_1_input],
        outputs=diff_sent_encoded,
    )

    bilbowa_model_infer = Model(
        inputs=[sent_0_input, mask_0_input], outputs=sent_0_encoded)

    return (
        word2vec_model,
        bilbowa_model,
        word2vec_model_infer,
        bilbowa_model_infer,
    )


def word2vec_loss(y_true, y_pred):
    # y_true is label (0 or 1)
    # y_pred is the dot prod

    # 0 / 1 -> 1. -> -1.
    a = (K.cast(y_true, dtype='float32') * 2 - 1.0) * (-1.0)
    return K.softplus(a * y_pred)


def bilbowa_loss(y_true, y_pred):
    # y_true is dummy here
    diff_sent_encoded = y_pred
    return K.mean(K.square(diff_sent_encoded), axis=-1)