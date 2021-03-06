#!/usr/bin/env python3

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Multiply,Dense, Activation, Dropout, Embedding, LSTM, CuDNNGRU, Bidirectional, BatchNormalization, merge, Conv1D, Dot, Multiply, Lambda, Subtract, TimeDistributed
from keras.layers.core import Flatten, Reshape
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.layers import Input
import tensorflow as tf



from keras.initializers import RandomNormal
from keras.layers.merge import dot
from keras.optimizers import TFOptimizer
from keras.utils import plot_model


config = tf.ConfigProto( device_count = {'GPU': 4 , 'CPU': 20} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def get_model(
        nb_word,
        dim,
        length,
        desc_length,
        s_negative_samples,
        w_negative_samples,
        word_emb_matrix=None,
        context_emb_matrix=None,
        word_emb_trainable=True,
        context_emb_trainable=True
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
        # return diff_sent_encoded

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


    # strong Pair model
    l0_s = Input(shape=(1,))
    l1_s = Input(shape=(1,))
    l0_s_n_lists = Input(shape=(s_negative_samples,))
    l1_s_n_lists = Input(shape=(s_negative_samples,))


    l0_s_embedded = word_emb(l0_s)
    l1_s_embedded = word_emb(l1_s)
    l0_s_n_lists_embedded = word_emb(l0_s_n_lists)
    l1_s_n_lists_embedded = word_emb(l1_s_n_lists)


    def l2_dist(x):
        y_true, y_pred = x
        l2 = K.mean(K.square(y_true - y_pred), axis=-1, keepdims=True)
        l2 = Flatten()(l2)
        l2 = K.exp(-l2)
        return l2

    def l2_dist_sum(x):
        '''
        y_true_list: batch_size x 10 x embedding_dim
        y_pred_list:
        '''
        y_true_list, y_pred_list = x
        l2 = K.mean(K.square(y_true_list - y_pred_list), axis=-1)
        l2_e = K.exp(-l2)
        print("l2_e", l2_e)
        l2 = K.sum(l2_e, axis = -1)
        return l2

    l2_dist_s_encode = Lambda(l2_dist)([
        l0_s_embedded,
        l1_s_embedded
    ])

    l2_dist_s_n_encode = Lambda(l2_dist_sum)([
        l0_s_n_lists_embedded,
        l1_s_n_lists_embedded
    ])

    print("s_negative_samples", s_negative_samples)


    def sub_loss(x):
        pos, neg = x
        loss = K.log(pos / neg)
        return loss

    def sub_loss_neg0(x):
        pos = x
        loss = K.log(pos)
        return loss

    if (s_negative_samples == 0):
        output_s = Lambda(sub_loss_neg0)([
            l2_dist_s_encode
        ])
    else:
        output_s = Lambda(sub_loss)([
            l2_dist_s_encode,
            l2_dist_s_n_encode
        ])

    strong_pair_model = Model(inputs=[l0_s, l1_s, l0_s_n_lists, l1_s_n_lists], outputs=output_s)

    # infer
    strong_pair_model_infer = Model(
        inputs=[l0_s], outputs=Flatten()(l0_s_embedded))


    # weak pair mode
    l0_w = Input(shape=(1,))
    l1_w = Input(shape=(1,))
    l0_w_n_lists = Input(shape=(w_negative_samples,))
    l1_w_n_lists = Input(shape=(w_negative_samples,))

    l0_w_embedded = word_emb(l0_w)
    l1_w_embedded = word_emb(l1_w)
    l0_w_n_lists_embedded = word_emb(l0_w_n_lists)
    l1_w_n_lists_embedded = word_emb(l1_w_n_lists)

    l2_dist_w_encode = Lambda(l2_dist)([
        l0_w_embedded,
        l1_w_embedded
    ])

    l2_dist_w_n_encode = Lambda(l2_dist_sum)([
        l0_w_n_lists_embedded,
        l1_w_n_lists_embedded
    ])

    print("w_negative_samples", w_negative_samples)

    if (w_negative_samples == 0):
        output_w = Lambda(sub_loss_neg0)([
            l2_dist_w_encode
        ])
    else:
        output_w = Lambda(sub_loss)([
            l2_dist_w_encode,
            l2_dist_w_n_encode
        ])

    weak_pair_model = Model(inputs=[l0_w, l1_w, l0_w_n_lists, l1_w_n_lists], outputs=output_w)

    # infer
    weak_pair_model_infer = Model(
        inputs=[l0_w], outputs=Flatten()(l0_w_embedded))

    return (
        word2vec_model,
        bilbowa_model,
        strong_pair_model,
        weak_pair_model,
        word2vec_model_infer,
        bilbowa_model_infer,
        strong_pair_model_infer,
        weak_pair_model_infer,
        word_emb,
        context_emb,
        diff_sent_encoded


    )


def word2vec_loss(y_true, y_pred):
    # 0 / 1 -> 1. -> -1.
    a = (K.cast(y_true, dtype='float32') * 2 - 1.0) * (-1.0)
    return K.softplus(a * y_pred)


def bilbowa_loss(y_true, y_pred):
    # y_true is dummy here
    diff_sent_encoded = y_pred
    return K.mean(K.square(diff_sent_encoded), axis=-1)

def strong_pair_loss(y_true, y_pred):
    return 0.7 * (-1) * y_pred
    # return 0.7 * (-1) * K.log(y_pred)


def weak_pair_loss(y_true, y_pred):
    return 0.4 * (-1) * y_pred


