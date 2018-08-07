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

from keras.optimizers import Adam

# from data import Embedding, MultiLanguageEmbedding, \
#     LazyIndexCorpus,  Word2vecIterator, BilbowaIterator

from data import *
from model import get_model, word2vec_loss, bilbowa_loss, strong_pair_loss, weak_pair_loss
sys.path.insert(0, '../eval')
from evaluate import Evaluator
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU': 5 , 'CPU': 56} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)


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
flags.DEFINE_float('word2vec_lr', 0.001, '(Negative for default)')
flags.DEFINE_integer('bilbowa_sent_length', 50, '')
flags.DEFINE_integer('bilbowa_batch_size', 100, '')
flags.DEFINE_float('bilbowa_lr', 0.001, '(Negative for default)')
flags.DEFINE_integer('encoder_desc_length', 15, '')
flags.DEFINE_integer('encoder_batch_size', 50, '')
flags.DEFINE_float('encoder_lr', 0.0002, '')

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

    emb0 = Embedding(join(FLAGS.data_root, FLAGS.lang0_emb_file))
    emb0_size = len(emb0.vocab)
    emb1 = Embedding(join(FLAGS.data_root, FLAGS.lang1_emb_file))
    emb1_size = len(emb1.vocab)
    emb = MultiLanguageEmbedding(emb0, emb1)
    vocab = emb.get_vocab()
    emb_matrix = emb.get_emb()

    # results = evaluator.word_translation()

    strong, weak = read_pair()
    strong_id, weak_id, l0_dict, l1_dict = pair2id(strong, weak, emb)

    ctxemb0 = Embedding(join(FLAGS.data_root, FLAGS.lang0_ctxemb_file))
    ctxemb1 = Embedding(join(FLAGS.data_root, FLAGS.lang1_ctxemb_file))
    ctxemb = MultiLanguageEmbedding(ctxemb0, ctxemb1)
    ctxvocab = ctxemb.get_vocab()
    ctxemb_matrix = ctxemb.get_emb()

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

    # strong pair iterator
    strong_batch_size = 1000
    strong_negative_size = 0
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

    (
        word2vec_model,
        bilbowa_model,
        strong_pair_model,
        weak_pair_model,
        word2vec_model_infer,
        bilbowa_model_infer,
        strong_pair_model_infer,
        weak_pair_model_infer,
        word_emb
    ) = get_model(
        nb_word=len(vocab),
        dim=FLAGS.emb_dim,
        length=FLAGS.bilbowa_sent_length,
        desc_length=FLAGS.encoder_desc_length,
        word_emb_matrix=emb_matrix,
        context_emb_matrix=ctxemb_matrix,
    ) #emb_matrix

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
    if FLAGS.train_mono:
        keys.append('mono0')
        keys.append('mono1')
    # if FLAGS.train_multi:
    keys.append('multi')
    keys.append('strong_pair')
    keys.append('weak_pair')
    keys = tuple(keys)

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
            this_load_time = time.time() - start_time
            start_time = time.time()
            # print("mono1", x)
            loss = word2vec_model.train_on_batch(x=x, y=y)
            this_comp_time = time.time() - start_time - 0.1
        elif next_key == 'multi':
            start_time = time.time()
            (x, y), (epoch, instance) = next(multi_iter)
            this_load_time = time.time() - start_time
            start_time = time.time()
            loss = bilbowa_model.train_on_batch(x=x, y=y)
            this_comp_time = time.time() - start_time
        elif next_key == 'strong_pair':
            start_time = time.time()
            (x, y), (epoch, instance) = next(strong_iter)
            this_load_time = time.time() - start_time
            start_time = time.time()
            loss = strong_pair_model.train_on_batch(x=x, y=y)
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
                           FLAGS.logging_iterval):
            last_logging_time = total_this_comp_time
            # logging.info('Stats so far')
            # logging.info('next_key = %s', next_key)
            # logging.info('comp_time = %s', dict_to_str(comp_time))
            # logging.info('load_time = %s', dict_to_str(load_time))
            # logging.info('total_time = %s', dict_to_str(get_total_time()))
            # logging.info('hit_count = %s', dict_to_str(hit_count))
            logging.info('iter_info = %s', dict_to_str(iter_info))
            print(next_key)
            logging.info('last_loss = %s', dict_to_str(last_loss))


        if should_exit or (total_this_comp_time - last_eval_time >
                           100):
            last_eval_time = total_this_comp_time
            # evaluate:
            if (next_key == 'mono1' or next_key == 'mono0'):
                pass
            else:
                word_emb_np = word_emb.get_weights()[0]
                embedding0 = word_emb_np[0:emb0_size,:]
                embedding1 = word_emb_np[emb0_size:,:] # 995003, 39016
                # muse test set
                print("en-fr_test")
                evaluator = Evaluator(embedding0,embedding1, emb0.vocablower2id, emb1.vocablower2id, "en", "fr", "default")
                results = evaluator.word_translation()

                # fr - en
                print("fr-en_test")
                evaluator = Evaluator(embedding1, embedding0, emb1.vocablower2id, emb0.vocablower2id, "fr", "en", "default")
                results = evaluator.word_translation()

                # strong weak pair set
                print("strong_test")
                evaluator = Evaluator(embedding1, embedding0, emb1.vocablower2id, emb0.vocablower2id, "fr", "en", "strong")
                results = evaluator.word_translation()


        # save model
        if should_exit or (total_this_comp_time - last_saving_time >
                           FLAGS.saving_iterval):
            last_saving_time = total_this_comp_time
            logging.info('Saving Embedding started.')
            # tag = ''
            # word2vec_model.save(join(FLAGS.model_root, tag + 'word2vec_model'))
            # bilbowa_model.save(join(FLAGS.model_root, tag + 'bilbowa_model'))
            # word2vec_model_infer.save(
            #     join(FLAGS.model_root, tag + 'word2vec_model_infer'))
            # bilbowa_model_infer.save(
            #     join(FLAGS.model_root, tag + 'bilbowa_model_infer'))

            # save embedding:
            word_emb_np = word_emb.get_weights()[0]
            emb0_save = word_emb_np[0:emb0_size, :]
            emb0_vocab = np.array(emb0.vocab)
            with open('./save_embed/random_withctx.en-fr.en.50.1.txt', 'w') as f:
                f.write(str(emb0_size))
                f.write(' ')
                f.write("50")
                f.write('\n')
                for name, vector in zip(emb0_vocab, emb0_save):
                    # f.write(name)
                    # f.write(' ')
                    np.savetxt(f, vector, fmt='%.6f', newline=" ")
                    f.write('\n')

            emb1_save = word_emb_np[emb0_size:, :]
            emb1_vocab = np.array(emb1.vocab)
            with open('./save_embed/random_withctx.en-fr.fr.50.1.txt', 'w',errors='ignore') as f:
                f.write(str(emb1_size))
                f.write(' ')
                f.write("50")
                f.write('\n')
                for name, vector in zip(emb1_vocab, emb1_save):
                    # f.write(name)
                    # f.write(' ')
                    np.savetxt(f, vector, fmt='%.6f', newline=" ")
                    f.write('\n')

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
