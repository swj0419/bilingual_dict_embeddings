#!/usr/bin/env python3

import math
import os
from os.path import join
import time
import sys

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import numpy as np
import pickle
import datetime

from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


from data import Embedding, MultiLanguageEmbedding, \
    LazyIndexCorpus,  Word2vecIterator, BilbowaIterator
import os

# config = tf.ConfigProto()
# session = tf.Session(config=config)
# KTF.set_session(session)


from data import *
from model import get_model, word2vec_loss, bilbowa_loss, strong_pair_loss, weak_pair_loss
sys.path.insert(0, '../eval')
from evaluate import Evaluator
import tensorflow as tf
import keras



FLAGS = flags.FLAGS

# data related
flags.DEFINE_string('data_root', '', 'root directory for data.')
flags.DEFINE_string('lang0_emb_file', '', '')
flags.DEFINE_string('lang1_emb_file', '', '')
flags.DEFINE_string('lang0_ctxemb_file', '', '')
flags.DEFINE_string('lang1_ctxemb_file', '', '')
flags.DEFINE_string('lang01_desc_file', '', '')
flags.DEFINE_string('lang10_desc_file', '', '')

flags.DEFINE_integer('mono_max_lines', -1, '')
flags.DEFINE_integer('multi_max_lines', -1, '')

# model related (also data)
flags.DEFINE_string('model_root', '', 'root directory for model')
flags.DEFINE_string('lang0_mono_index_corpus_file', '', '')
flags.DEFINE_string('lang1_mono_index_corpus_file', '', '')
flags.DEFINE_string('lang0_multi_index_corpus_file', '', '')
flags.DEFINE_string('lang1_multi_index_corpus_file', '', '')

# training related
flags.DEFINE_integer('emb_dim', 50, '')
flags.DEFINE_float('emb_subsample', 1e-5, '')
flags.DEFINE_integer('word2vec_negative_size', 10, '')
flags.DEFINE_integer('word2vec_batch_size', 100000, '')
flags.DEFINE_float('word2vec_lr', 0.00001, '(Negative for default)')
flags.DEFINE_integer('bilbowa_sent_length', 60, '')
flags.DEFINE_integer('bilbowa_batch_size', 100, '')
flags.DEFINE_float('bilbowa_lr', 0.00001, '(Negative for default)')
flags.DEFINE_integer('encoder_desc_length', 15, '')
flags.DEFINE_integer('encoder_batch_size', 20, '')
flags.DEFINE_float('encoder_lr', 0.02, '')

flags.DEFINE_boolean('train_mono', True, '')
flags.DEFINE_boolean('train_multi', True, '')
flags.DEFINE_integer('max_mono_epochs', -1, '')
flags.DEFINE_integer('max_multi_epochs', -1, '')
flags.DEFINE_boolean('word_emb_trainable', True, '')
flags.DEFINE_boolean('context_emb_trainable', True, '')
flags.DEFINE_boolean('encoder_target_no_gradient', True, '')
flags.DEFINE_boolean('encoder_arch_version', 1, '')

flags.DEFINE_float('logging_iterval', 5, '')
flags.DEFINE_float('saving_iterval', 500, '')


def main(argv):
    del argv  # Unused.
    os.system('mkdir -p "%s"' % FLAGS.model_root)
    emb0 = pickle.load(open("./sav_model/pretrained/emb0.pickle", "rb", -1))
    # emb0 = Embedding(join(FLAGS.data_root, FLAGS.lang0_emb_file))
    emb0_size = len(emb0.vocab)
    emb1 = pickle.load(open("./sav_model/pretrained/emb1.pickle", "rb", -1))
    # emb1 = Embedding(join(FLAGS.data_root, FLAGS.lang1_emb_file))
    emb1_size = len(emb1.vocab)
    emb = MultiLanguageEmbedding(emb0, emb1)
    vocab = emb.get_vocab()
    emb_matrix = emb.get_emb()

    ctxemb0 = pickle.load(open("./sav_model/pretrained/emb0_cxt.pickle", "rb", -1))
    # ctxemb0 = Embedding(join(FLAGS.data_root, FLAGS.lang0_ctxemb_file))
    ctxemb1 = pickle.load(open("./sav_model/pretrained/emb1_cxt.pickle", "rb", -1))
    # ctxemb1 = Embedding(join(FLAGS.data_root, FLAGS.lang1_ctxemb_file))
    ctxemb = MultiLanguageEmbedding(ctxemb0, ctxemb1)
    ctxvocab = ctxemb.get_vocab()
    ctxemb_matrix = ctxemb.get_emb()
    logging.info('load embedding done')



    strong, weak = read_pair()
    strong_id, weak_id, l0_dict, l1_dict = pair2id(strong, weak, emb)

    assert tuple(ctxvocab) == tuple(vocab)

    mono0 = LazyIndexCorpus(
        join(FLAGS.data_root, FLAGS.lang0_mono_index_corpus_file),
        max_lines=FLAGS.mono_max_lines)

    mono1 = LazyIndexCorpus(
        join(FLAGS.data_root, FLAGS.lang1_mono_index_corpus_file),
        max_lines=FLAGS.mono_max_lines)

    multi0 = LazyIndexCorpus(
        join(FLAGS.data_root, FLAGS.lang0_multi_index_corpus_file),
        max_lines=FLAGS.multi_max_lines)

    multi1 = LazyIndexCorpus(
        join(FLAGS.data_root, FLAGS.lang1_multi_index_corpus_file),
        max_lines=FLAGS.multi_max_lines)

    mono0_unigram_table = mono0.get_unigram_table(vocab_size=len(vocab))
    mono1_unigram_table = mono1.get_unigram_table(vocab_size=len(vocab))

    mono0_iterator = Word2vecIterator(
        mono0,
        mono0_unigram_table,
        subsample=FLAGS.emb_subsample,
        window_size=FLAGS.word2vec_negative_size,
        negative_samples=FLAGS.word2vec_negative_size,
        batch_size=FLAGS.word2vec_batch_size
    )

    mono1_iterator = Word2vecIterator(
        mono1,
        mono1_unigram_table,
        subsample=FLAGS.emb_subsample,
        window_size=FLAGS.word2vec_negative_size,
        negative_samples=FLAGS.word2vec_negative_size,
        batch_size=FLAGS.word2vec_batch_size
    )

    multi_iterator = BilbowaIterator(
        multi0,
        multi1,
        mono0_unigram_table,
        mono1_unigram_table,
        subsample=FLAGS.emb_subsample,
        length=FLAGS.bilbowa_sent_length,
        batch_size=FLAGS.bilbowa_batch_size
    )

    # multi_iterator.logging_debug(emb)

    # strong pair iterator
    strong_batch_size = 1000
    strong_negative_size = 5
    strong_pair_iterator = strong_pairIterator(
        strong_id,
        mono0_unigram_table,
        mono1_unigram_table,
        batch_size = strong_batch_size,
        negative_samples = strong_negative_size,
        l0_dict = l0_dict,
        l1_dict = l1_dict
    )

    # weak pair iterator
    weak_batch_size = 3000
    weak_negative_size = 0
    weak_pair_iterator = weak_pairIterator(
        weak_id,
        mono0_unigram_table,
        mono1_unigram_table,
        batch_size=weak_batch_size,
        negative_samples=weak_negative_size,
        l0_dict=l0_dict,
        l1_dict=l1_dict
    )

    word_emb_matrix = emb_matrix
    context_emb_matrix = ctxemb_matrix #emb_matrix, ctxemb_matrix
    (
        word2vec_model,
        bilbowa_model,
        strong_pair_model,
        weak_pair_model,
        word2vec_model_infer,
        bilbowa_model_infer,
        strong_pair_model_infer,
        weak_pair_model_infer,
        word_emb,
        word_emb_cxt,
        diff_sent_encoded
    ) = get_model(
        nb_word=len(vocab),
        dim=FLAGS.emb_dim,
        length=FLAGS.bilbowa_sent_length,
        desc_length=FLAGS.encoder_desc_length,
        s_negative_samples=strong_negative_size,
        w_negative_samples=weak_negative_size,
        word_emb_matrix=word_emb_matrix,
        context_emb_matrix=context_emb_matrix
    ) #emb_matrix, ctxemb_matrix


    logging.info('word2vec_model.summary()')
    word2vec_model.summary()
    logging.info('bilbowa_model.summary()')
    bilbowa_model.summary()
    logging.info('strong_pair_model.summary()')
    strong_pair_model.summary()

    word2vec_model.compile(
        optimizer=(Adam(amsgrad=True) if FLAGS.word2vec_lr < 0 else Adam(
            lr=FLAGS.word2vec_lr, amsgrad=True)),
        loss=word2vec_loss)
    bilbowa_model.compile(
        optimizer=(Adam(amsgrad=True) if FLAGS.bilbowa_lr < 0 else Adam(
            lr=FLAGS.bilbowa_lr, amsgrad=True)),
        loss=bilbowa_loss)
    strong_pair_model_lr = 0.001
    strong_pair_model.compile(
        optimizer=(Adam(amsgrad=True) if strong_pair_model_lr < 0 else Adam(
            lr=strong_pair_model_lr, amsgrad=True)),
        loss=strong_pair_loss)

    weak_pair_model_lr = 0.001
    weak_pair_model.compile(
        optimizer=(Adam(amsgrad=True) if weak_pair_model_lr < 0 else Adam(
            lr=weak_pair_model_lr, amsgrad=True)),
        loss=weak_pair_loss)

    mono0_iter = mono0_iterator.fast2_iter()
    mono1_iter = mono1_iterator.fast2_iter()
    multi_iter = multi_iterator.iter()
    strong_iter = strong_pair_iterator.strong_iter()
    weak_iter = weak_pair_iterator.weak_iter()

    # weak
    keys = []
    # if FLAGS.train_mono:
    # keys.append('mono0')
    # keys.append('mono1')
    # keys.append('multi')
    keys.append('strong_pair')
    keys.append('weak_pair')
    keys = tuple(keys)


    '''
    Save Model
    '''
    def get_model_identifier(keys, dim, strong_negative_size, weak_negative_size):
        prefix = "dim" + str(dim) + "_"
        for i in keys:
            if(i == "mono0"):
                prefix += "mono_" + str(FLAGS.word2vec_lr) + "_"
            elif(i == "multi"):
                prefix += "multi_" + str(FLAGS.bilbowa_lr) + "_"
            elif(i == "strong_pair"):
                prefix = prefix + "s" + str(strong_negative_size) + "_" + str(strong_pair_model_lr) + "_"
            elif (i == "weak_pair"):
                prefix = prefix + "w" + str(weak_negative_size) + "_"  + str(weak_pair_model_lr) + "_"

        now = datetime.datetime.now()
        date = '%02d%02d' % (now.month, now.day)  # two digits month/day
        identifier = date + "_" + prefix
        if(word_emb_matrix is None):
            word_emb_matrix_str = "None"
        else:
            word_emb_matrix_str = "Pretrained"
        identifier = identifier + word_emb_matrix_str
        # identifier = '_'.join([prefix, date, str(lr), str(dim), str(batch_size), str(p_neg)])
        return identifier

    save_dir = "./sav_model"
    identifier = get_model_identifier(keys, FLAGS.emb_dim, strong_negative_size, weak_negative_size)
    save_dir = join(save_dir, identifier)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def save_emb_obj(word_emb_np, word_emb_cxt,save_dir):
        emb0_save = word_emb_np[0:emb0_size, :]
        emb1_save = word_emb_np[emb0_size:, :]
        emb0.emb = emb0_save
        emb1.emb = emb1_save
        embedding0_cxt = word_emb_cxt[0:emb0_size, :]
        embedding1_cxt = word_emb_cxt[emb0_size:, :]
        ctxemb0.emb = embedding0_cxt
        ctxemb1.emb = embedding1_cxt

        dir0 = join(save_dir, "emb0.pickle")
        dir1 = join(save_dir, "emb1.pickle")
        dir3= join(save_dir, "emb0_cxt.pickle")
        dir4 = join(save_dir, "emb1_cxt.pickle")
        with open(dir0, "wb") as file_:
            pickle.dump(emb0, file_, -1)
        with open(dir1, "wb") as file_:
            pickle.dump(emb1, file_, -1)
        with open(dir3, "wb") as file_:
            pickle.dump(ctxemb0, file_, -1)
        with open(dir4, "wb") as file_:
            pickle.dump(ctxemb1, file_, -1)


    '''
    Train Model
    '''
    def dict_to_str(d):
        return '{' + ', '.join(
            ['%s: %s' % (key, d[key]) for key in sorted(d.keys())]) + '}'


    comp_time = {key: 0.0 for key in keys}
    load_time = {key: 0.0 for key in keys}
    hit_count = {key: 0 for key in keys}
    iter_info = {key: (0, 0) for key in keys}
    last_loss = {key: 0.0 for key in keys}

    def get_total_time():
        return {key: comp_time[key] + load_time[key] for key in keys}

    global_start_time = time.time()
    last_logging_time = 0.
    loss_decay = 0.6
    last_saving_time = 0.
    last_eval_time = 0


    while True:
        total_time = get_total_time()
        target_time = total_time
        min_time = min(target_time.values())
        next_key = [key for key in keys if target_time[key] == min_time][0]

        if next_key == 'mono0':
            start_time = time.time()
            (x, y), (epoch, instance) = next(mono0_iter)
            # print("mono0", x)
            # print("mono0", y)
            this_load_time = time.time() - start_time
            start_time = time.time()
            loss = word2vec_model.train_on_batch(x=x, y=y)
            this_comp_time = time.time() - start_time
        elif next_key == 'mono1':
            start_time = time.time()
            (x, y), (epoch, instance) = next(mono1_iter)
            # print("x", x)
            this_load_time = time.time() - start_time
            start_time = time.time()
            loss = word2vec_model.train_on_batch(x=x, y=y)
            this_comp_time = time.time() - start_time
        elif next_key == 'multi':
            start_time = time.time()
            (x, y), (epoch, instance) = next(multi_iter)
            this_load_time = time.time() - start_time
            start_time = time.time()
            loss = bilbowa_model.train_on_batch(x=x, y=y)
            this_comp_time = time.time() - start_time
            # out = bilbowa_model.predict(x)
            # print("diff_sent_encoded", out)
            # loss_s = K.mean(K.square(out), axis=-1)
            # loss_t = K.mean(loss_s, axis=-1)
            # loss_s = K.eval(loss_s)
            # loss_t = K.eval(loss_t)
            # print("loss_s", loss_s)
            # print("loss_t", loss_t)

        elif next_key == 'strong_pair':
            start_time = time.time()
            (x, y), (epoch, instance) = next(strong_iter)
            this_load_time = time.time() - start_time
            start_time = time.time()
            loss = strong_pair_model.train_on_batch(x=x, y=y)
            # print(K.eval(output_s))
            # print("L2 DIST", strong_pair_model.predict(x))

            this_comp_time = time.time() - start_time
        elif next_key == 'weak_pair':
            start_time = time.time()
            (x, y), (epoch, instance) = next(weak_iter)
            this_load_time = time.time() - start_time
            start_time = time.time()
            loss = weak_pair_model.train_on_batch(x=x, y=y)
            this_comp_time = time.time() - start_time
        else:
            assert False

        assert not math.isnan(loss)


        comp_time[next_key] += this_comp_time
        load_time[next_key] += this_load_time
        hit_count[next_key] += 1
        iter_info[next_key] = (epoch, instance)
        last_loss[next_key] = loss if last_loss[next_key] == 0.0 else (
            last_loss[next_key] * loss_decay + loss * (1. - loss_decay))

        # exit if target is reached
        should_exit = False
        if FLAGS.max_mono_epochs > -1:
            if (iter_info['mono0'][0] >= FLAGS.max_mono_epochs
                    and iter_info['mono1'][0] >= FLAGS.max_mono_epochs):
                should_exit = True

        if FLAGS.max_multi_epochs > -1:
            if (iter_info['multi'][0] >= FLAGS.max_multi_epochs):
                should_exit = True
        total_this_comp_time = time.time() - global_start_time


        if should_exit or (total_this_comp_time - last_logging_time >
                           5): # FLAGS.logging_iterval
            last_logging_time = total_this_comp_time
            # logging.info('Stats so far')
            # logging.info('next_key = %s', next_key)
            # logging.info('comp_time = %s', dict_to_str(comp_time))
            # logging.info('load_time = %s', dict_to_str(load_time))
            # logging.info('total_time = %s', dict_to_str(get_total_time()))
            # logging.info('hit_count = %s', dict_to_str(hit_count))
            logging.info('iter_info = %s', dict_to_str(iter_info))
            logging.info('last_loss = %s', dict_to_str(last_loss))
            logname = join(save_dir, "loss.txt")
            with open(logname, "a") as f:
                f.write(dict_to_str(iter_info))
                f.write("\n")
                f.write(dict_to_str(last_loss))
                f.write("\n")


        if should_exit or (total_this_comp_time - last_eval_time >
                           250):
            last_eval_time = total_this_comp_time
            # evaluate:
            if (0): # next_key == 'mono1' or next_key == 'mono0'
                pass
            else:
                word_emb_np = word_emb.get_weights()[0]
                embedding0 = word_emb_np[0:emb0_size,:]
                embedding1 = word_emb_np[emb0_size:,:] # 995003, 39016
                # muse test set
                print("en-fr_test")
                evaluator = Evaluator(embedding0,embedding1, emb0.vocablower2id, emb1.vocablower2id, "en", "fr", "default")
                results = evaluator.word_translation()[0]
                logname = join(save_dir, "word_translation.txt")
                with open(logname, "a") as f:
                    f.write("en-fr_test: ")
                    f.write(str(results))
                    f.write("\n")

                # fr - en
                print("fr-en_test")
                evaluator = Evaluator(embedding1, embedding0, emb1.vocablower2id, emb0.vocablower2id, "fr", "en", "default")
                results = evaluator.word_translation()[0]
                with open(logname, "a") as f:
                    f.write("fr-en_test: ")
                    f.write(str(results))
                    f.write("\n")

                # strong weak pair set
                print("strong_test")
                evaluator = Evaluator(embedding1, embedding0, emb1.vocablower2id, emb0.vocablower2id, "fr", "en", "strong")
                results = evaluator.word_translation()[0]
                with open(logname, "a") as f:
                    f.write("strong_test: ")
                    f.write(str(results))
                    f.write("\n")
                    f.write("-----------------------------------------------------------")
                    f.write("\n")



        # save model
        if should_exit or (total_this_comp_time - last_saving_time >
                           250): #FLAGS.saving_iterval
            last_saving_time = total_this_comp_time
            logging.info('Saving Embedding started.')

            # save embedding:
            print('The model will be stored in: ', save_dir)
            word_emb_np = word_emb.get_weights()[0]
            word_emb_cxt_np = word_emb_cxt.get_weights()[0]

            save_emb_obj(word_emb_np,word_emb_cxt_np, save_dir)
            logging.info('Saving Embedding done.')

        if should_exit:
            logging.info('Training target reached. Exit.')
            break


import pdb, traceback, sys, code  # noqa
if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
