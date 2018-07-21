#!/usr/bin/env python3

import math
import os
from os.path import join
import time

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import numpy as np

from keras.optimizers import Adam

from data import Embedding, MultiLanguageEmbedding, \
    LazyIndexCorpus,  Word2vecIterator, BilbowaIterator
from model import get_model, word2vec_loss, bilbowa_loss

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
flags.DEFINE_float('word2vec_lr', -1., '(Negative for default)')
flags.DEFINE_integer('bilbowa_sent_length', 50, '')
flags.DEFINE_integer('bilbowa_batch_size', 100, '')
flags.DEFINE_float('bilbowa_lr', -1., '(Negative for default)')
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
flags.DEFINE_float('saving_iterval', 3600, '')


def main(argv):
    del argv  # Unused.

    os.system('mkdir -p "%s"' % FLAGS.model_root)

    emb0 = Embedding(join(FLAGS.data_root, FLAGS.lang0_emb_file))
    emb1 = Embedding(join(FLAGS.data_root, FLAGS.lang1_emb_file))
    emb = MultiLanguageEmbedding(emb0, emb1)
    vocab = emb.get_vocab()
    emb_matrix = emb.get_emb()

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
        batch_size=FLAGS.word2vec_batch_size,
    )
    mono1_iterator = Word2vecIterator(
        mono1,
        mono1_unigram_table,
        subsample=FLAGS.emb_subsample,
        window_size=FLAGS.word2vec_negative_size,
        negative_samples=FLAGS.word2vec_negative_size,
        batch_size=FLAGS.word2vec_batch_size,
    )
    multi_iterator = BilbowaIterator(
        multi0,
        multi1,
        mono0_unigram_table,
        mono1_unigram_table,
        subsample=FLAGS.emb_subsample,
        length=FLAGS.bilbowa_sent_length,
        batch_size=FLAGS.bilbowa_batch_size,
    )

    (
        word2vec_model,
        bilbowa_model,
        word2vec_model_infer,
        bilbowa_model_infer,
    ) = get_model(
        nb_word=len(vocab),
        dim=FLAGS.emb_dim,
        length=FLAGS.bilbowa_sent_length,
        desc_length=FLAGS.encoder_desc_length,
        word_emb_matrix=emb_matrix,
        context_emb_matrix=ctxemb_matrix,
    )

    logging.info('word2vec_model.summary()')
    word2vec_model.summary()
    logging.info('bilbowa_model.summary()')
    bilbowa_model.summary()

    word2vec_model.compile(
        optimizer=(Adam(amsgrad=True) if FLAGS.word2vec_lr < 0 else Adam(
            lr=FLAGS.word2vec_lr, amsgrad=True)),
        loss=word2vec_loss)
    bilbowa_model.compile(
        optimizer=(Adam(amsgrad=True) if FLAGS.bilbowa_lr < 0 else Adam(
            lr=FLAGS.bilbowa_lr, amsgrad=True)),
        loss=bilbowa_loss)

    mono0_iter = mono0_iterator.fast2_iter()
    mono1_iter = mono1_iterator.fast2_iter()
    multi_iter = multi_iterator.iter()

    keys = []
    if FLAGS.train_mono:
        keys.append('mono0')
        keys.append('mono1')
    if FLAGS.train_multi:
        keys.append('multi')
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

    while True:
        total_time = get_total_time()
        target_time = total_time
        min_time = min(target_time.values())
        next_key = [key for key in keys if target_time[key] == min_time][0]

        if next_key == 'mono0':
            start_time = time.time()
            (x, y), (epoch, instance) = next(mono0_iter)
            this_load_time = time.time() - start_time
            start_time = time.time()
            loss = word2vec_model.train_on_batch(x=x, y=y)
            this_comp_time = time.time() - start_time
        elif next_key == 'mono1':
            start_time = time.time()
            (x, y), (epoch, instance) = next(mono1_iter)
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
            logging.info('Stats so far')
            logging.info('next_key = %s', next_key)
            logging.info('comp_time = %s', dict_to_str(comp_time))
            logging.info('load_time = %s', dict_to_str(load_time))
            logging.info('total_time = %s', dict_to_str(get_total_time()))
            logging.info('hit_count = %s', dict_to_str(hit_count))
            logging.info('iter_info = %s', dict_to_str(iter_info))
            logging.info('last_loss = %s', dict_to_str(last_loss))

        if should_exit or (total_this_comp_time - last_saving_time >
                           FLAGS.saving_iterval):
            last_saving_time = total_this_comp_time
            logging.info('Saving models started.')
            tag = ''
            word2vec_model.save(join(FLAGS.model_root, tag + 'word2vec_model'))
            bilbowa_model.save(join(FLAGS.model_root, tag + 'bilbowa_model'))
            word2vec_model_infer.save(
                join(FLAGS.model_root, tag + 'word2vec_model_infer'))
            bilbowa_model_infer.save(
                join(FLAGS.model_root, tag + 'bilbowa_model_infer'))
            logging.info('Saving models done.')

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