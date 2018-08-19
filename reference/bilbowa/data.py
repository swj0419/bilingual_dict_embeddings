#!/usr/bin/env python3

import csv
from collections import namedtuple
from itertools import chain
import math
from multiprocessing import Process, Queue
import random

from joblib import Parallel, delayed
import numpy as np
import numpy
from tqdm import tqdm
import yaml

from absl import app
from absl import flags
from absl import logging
from chainer.backends import cuda





class Embedding(object):
    def __init__(self, emb_file='', stopwords_file='', keep_emb=True):
        self.vocab = []
        self.emb = []
        self.vocab2id = dict()
        self.vocablower2id = dict()
        self.id2vocablower = dict()
        self.stopwords = set()


        if emb_file:
            (
                self.vocab,
                self.emb,
                self.vocab2id,
                self.vocablower2id,
                self.id2vocablower

            ) = self.load_emb(
                emb_file, keep_emb=keep_emb)

        if stopwords_file:
            self.stopwords = self.load_stopwords(stopwords_file)

        self.stopwords = (self.load_stopwords(stopwords_file)
                          if stopwords_file else set())

        if not keep_emb:
            self.emb = None
        logging.info('Embedding: initialized')

    def __len__(self):
        return len(self.vocab)

    def load_emb(self, filepath, keep_emb=True):
        logging.info('Embedding: Loading embedding from %s', filepath)
        vocab, emb = [], []
        fin = open(filepath, errors='surrogateescape')
        n, dim = map(int, fin.readline().strip().split())
        count = 0
        for i in tqdm(range(n), unit=' words'):
            tokens = fin.readline().strip().split()
            if len(tokens) == dim + 1:
                vocab.append(tokens[0])
                if keep_emb:
                    emb.append(tokens[1:])
            # not load all embedding
            # count += 1
            # if(count == 20000):
            #     break

        emb = np.array(emb, dtype='f')
        fin.close()

        logging.info('Embedding: Making mapping')
        vocab2id = {w: i for i, w in enumerate(vocab)}
        vocablower2id = {w.lower(): i for i, w in enumerate(vocab)}
        id2vocablower = {i: w.lower() for i, w in enumerate(vocab)}
        return vocab, emb, vocab2id, vocablower2id, id2vocablower

    def load_stopwords(self, filepath):
        stop_words = []
        for line in open(filepath):
            stop_words.append(line.strip())
        return set(stop_words)

    def get_emb(self):
        return self.emb

    def encode(self, word_or_words, offset=0, lang_id=0):
        assert lang_id == 0

        if isinstance(word_or_words, (list, tuple)):
            words = word_or_words

            return [self.encode(word, offset, lang_id) for word in words]
        else:
            word = word_or_words
            word_lower = word.lower()

            id_ = None
            if word in self.vocab2id and word not in self.stopwords:
                id_ = self.vocab2id[word]
            elif (word_lower in self.vocab2id
                  and word_lower not in self.stopwords):
                id_ = self.vocab2id[word_lower]
            elif (word_lower in self.vocablower2id
                  and word_lower not in self.stopwords):
                id_ = self.vocablower2id[word_lower]

            if id_ is not None:
                return id_ + offset
            else:
                return -1


class MultiLanguageEmbedding(object):
    def __init__(self, *args):
        self.emb = args

    def __len__(self):
        return sum([len(_) for _ in self.emb])

    def encode(self, word_or_words, lang_id=0):
        assert 0 <= lang_id < len(self.emb)
        offset = sum([len(self.emb[id_]) for id_ in range(lang_id)])
        result = self.emb[lang_id].encode(word_or_words, offset=offset)
        return result

    def get_vocab(self):
        result = []
        for _ in self.emb:
            result.extend(_.vocab)
        return result

    def get_emb(self):
        return np.concatenate(tuple([_.emb for _ in self.emb]), axis=0)

    def get_dim(self):
        return self.emb[0].emb.shape[1]


def iter_as_batch(it, batch_size=1):
    batch = []
    for elem in it:
        batch.append(elem)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch
        batch = []


def line_to_id(line, emb, lang_id):
    result = emb.encode(line.strip().split() + ['</s>'], lang_id=lang_id)
    result = [_ for _ in result if _ > -1]
    return result


def lines_to_id(lines, emb, lang_id):
    result = [line_to_id(line, emb, lang_id) for line in lines]
    return result


def subsample(ids, counts, total_words, sample):

    product_sample_totoal_words = sample * total_words

    def kill_word(count):
        threshold = (math.sqrt(count * 1.0 / (product_sample_totoal_words)) +
                     1) * (product_sample_totoal_words) / (1.0 * count)
        return threshold < random.random()

    def process_row(row):
        use_np = True
        if use_np:
            row_counts = counts[row] * 1.0
            row_counts = row_counts + 1.0  # TODO: temporary fix

            threshold = (np.sqrt(row_counts / product_sample_totoal_words) +
                         1.0) * (product_sample_totoal_words) / row_counts
            to_kill = threshold < np.random.random(size=threshold.shape)
            result = [
                id_ for id_, _to_kill in zip(row, to_kill) if not _to_kill
            ]
        else:
            result = [id_ for id_ in row if not kill_word(counts[id_] + 1)]
        return result

    return [process_row(row) for row in ids]


def iter_word2vec_batched_func(kwargs, batched_raw_line):
    counts = kwargs['counts']
    counts_sum = kwargs['counts_sum']  # counts.sum()
    sample = kwargs['sample']
    window_size = kwargs['window_size']
    negative_samples = kwargs['negative_samples']
    negative_sampler = kwargs['negative_sampler']

    batched_row = []
    for raw_line in batched_raw_line:
        row = list(map(int, raw_line.strip().split()))
        row = [_ for _ in row if _ != -1]
        batched_row.append(row)

    subsampled_batched_row = subsample(
        batched_row,
        counts,
        counts_sum,
        sample,
    )

    positive_w_c_pair_list = []
    # special (w, -1) pair for place holder where negative example exists

    is_new_instance = []

    for row in subsampled_batched_row:
        row_length = len(row)
        for i in range(row_length):

            window_start = max(0, i - window_size)
            window_end = min(row_length, i + window_size + 1)

            mark = True
            for j in range(window_start, window_end):
                is_new_instance.append(1 if mark else 0)
                mark = False
                w = row[j]
                c = row[i] if i != j else -1
                positive_w_c_pair_list.append((w, c))

    negative_c_list_as_array = negative_sampler.sample(
        negative_samples * len(positive_w_c_pair_list))

    return positive_w_c_pair_list, is_new_instance, negative_c_list_as_array


def iter_word2vec_batched_func_worker(kwargs, qin, qout):
    while True:
        cmd, input_ = qin.get()
        if cmd == -1:
            break

        result = iter_word2vec_batched_func(kwargs, input_)
        qout.put(result)


class LazyIndexCorpus(object):
    def __init__(
            self,
            filepath,
            max_lines=-1,
            iter_async_allowed=False,
            iter_async_nb_works=1,
            iter_async_look_ahead=80,
            iter_batch_size=100,
    ):
        meta = yaml.load(open(filepath + '.meta.yaml'))
        self.eos_id = meta['eos_id']

        counts = np.load(filepath + '.counts.npz')
        self.counts = counts['counts']

        self.ids_filepath = filepath + '.ids.txt'

        self.max_lines = max_lines
        self.iter_async_allowed = iter_async_allowed
        self.iter_async_nb_works = iter_async_nb_works ###???
        self.iter_async_look_ahead = iter_async_look_ahead
        self.iter_batch_size = iter_batch_size

    def get_unigram_table(self, power=0.75, vocab_size=-1):
        counts = self.counts
        if vocab_size > -1 and len(counts) < vocab_size:
            counts = np.append(counts, [0] * (vocab_size - len(counts)))
        return UnigramTable(counts=counts, power=power)

    def iter_batched_raw_line(self, batch_size=1):
        with open(self.ids_filepath) as fin:
            batch = []
            for index, raw_line in enumerate(fin):
                if self.max_lines > 0 and index >= self.max_lines:
                    break
                batch.append(raw_line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch
                batch = []

    def iter_raw_line(self):
        buffer_size = 1000
        for batch in self.iter_batched_raw_line(buffer_size):
            for _ in batch:
                yield _

    def iter_word2vec(
            self,
            counts,
            counts_sum,
            sample,
            window_size,
            negative_samples,
            negative_sampler,
    ):
        kwargs = {
            'counts': counts,
            'counts_sum': counts_sum,
            'sample': sample,
            'window_size': window_size,
            'negative_samples': negative_samples,
            'negative_sampler': negative_sampler,
        }
        if self.iter_async_allowed:
            nb_proc = self.iter_async_nb_works
            batche_size = self.iter_batch_size
            look_ahead = self.iter_async_look_ahead

            qin = Queue()
            qout = Queue()

            procs = [
                Process(
                    target=iter_word2vec_batched_func_worker,
                    args=(kwargs, qin, qout)) for _ in range(nb_proc)
            ]

            for p in procs:
                p.start()

            i = 0
            for batched_raw_line in self.iter_batched_raw_line(batche_size):
                if i >= look_ahead:
                    yield qout.get()
                qin.put((0, batched_raw_line))
                i += 1

            for _ in range(look_ahead):
                qin.put((-1, None))
                yield qout.get()

            for p in procs:
                p.join()

        else:
            batche_size = self.iter_batch_size
            for batched_raw_line in self.iter_batched_raw_line(batche_size):
                batched_result = iter_word2vec_batched_func(
                    kwargs, batched_raw_line)
                yield batched_result

    def iter_sampled_ids(
            self,
            counts,
            sample=1e-5,
    ):
        counts_sum = counts.sum()
        for raw_line in self.iter_raw_line():
            row = list(map(int, raw_line.strip().split()))
            row = [_ for _ in row if _ != -1]
            sampled_bacthed_ids = subsample(
                [row],
                counts,
                counts_sum,
                sample,
            )[0]
            yield sampled_bacthed_ids


# Copied and modifed from
# https://github.com/chainer/chainer/blob/v4.0.0/chainer/utils/walker_alias.py
class WalkerAlias(object):
    """Implementation of Walker's alias method.
    This method generates a random sample from given probabilities
    :math:`p_1, \\dots, p_n` in :math:`O(1)` time.
    It is more efficient than :func:`~numpy.random.choice`.
    This class works on both CPU and GPU.
    Args:
        probs (float list): Probabilities of entries. They are normalized with
                            `sum(probs)`.
    See: `Wikipedia article <https://en.wikipedia.org/wiki/Alias_method>`_
    """

    def __init__(self, probs):
        prob = np.array(probs, np.float32)
        prob /= np.sum(prob)
        threshold = np.ndarray(len(probs), np.float32)
        values = np.ndarray(len(probs) * 2, np.int32)
        il, ir = 0, 0
        pairs = list(zip(prob, range(len(probs))))
        pairs.sort()

        for prob, i in pairs:
            p = prob * len(probs)
            while p > 1 and ir < il:
                values[ir * 2 + 1] = i
                p -= 1.0 - threshold[ir]
                ir += 1
            threshold[il] = p
            values[il * 2] = i
            il += 1
        # fill the rest
        for i in range(ir, len(probs)):
            values[i * 2 + 1] = 0

        assert ((values < len(threshold)).all())
        self.threshold = threshold
        self.values = values


    def sample(self, shape):
        """Generates a random sample based on given probabilities.
        Args:
            shape (tuple of int): Shape of a return value.
        Returns:
            Returns a generated array with the given shape. the return value
            is a :class:`numpy.ndarray` object.
        """

        return self.sample_cpu(shape)

    def sample_cpu(self, shape):
        ps = np.random.uniform(0, 1, shape)
        pb = ps * len(self.threshold)
        index = pb.astype(np.int32)
        left_right = (self.threshold[index] < pb - index).astype(np.int32)
        return self.values[index * 2 + left_right]


    def sample_gpu(self, shape):
        ps = cuda.cupy.random.uniform(size=shape, dtype=numpy.float32)
        vs = cuda.elementwise(
            'T ps, raw T threshold , raw S values, int32 b',
            'int32 vs',
            '''
            T pb = ps * b;
            int index = __float2int_rd(pb);
            // fill_uniform sometimes returns 1.0, so we need to check index
            if (index >= b) {
              index = 0;
            }
            int lr = threshold[index] < pb - index;
            vs = values[index * 2 + lr];
            ''',
            'walker_alias_sample'
        )(ps, self.threshold, self.values, len(self.threshold))
        return vs

class UnigramTable(object):
    def __init__(self, counts, power=0.75):
        self.counts = np.array(counts)
        power = np.float32(power)
        p = np.array(counts, power.dtype)
        np.power(p, power, p)
        self.negative_sampler = WalkerAlias(p)


InstanceInfo = namedtuple('InstanceInfo', ['batch_id', 'instance_id'])


# Some code copied and modified from
# https://github.com/ozgurdemir/word2vec-keras/blob/master/src/skip_gram.pyx
class Word2vecIterator(object):
    def __init__(self,
                 index_corpus,
                 unigram_table,
                 subsample,
                 window_size,
                 negative_samples,
                 batch_size,
                 epochs=-1):
        self.index_corpus = index_corpus
        self.unigram_table = unigram_table

        self.subsample = subsample
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.batch_size = batch_size
        self.epochs = epochs


    def fast2_iter(self, epochs=None):
        batch_size = self.batch_size
        epochs = self.epochs if epochs is None else epochs

        subsample = self.subsample
        window_size = self.window_size
        negative_samples = self.negative_samples
        counts = self.unigram_table.counts
        counts_sum = counts.sum()
        negative_sampler = self.unigram_table.negative_sampler

        DTYPE = np.int32
        words = np.zeros(shape=batch_size, dtype=DTYPE)
        contexts = np.zeros(shape=batch_size, dtype=DTYPE)
        labels = np.empty(shape=batch_size, dtype=DTYPE)

        pointer = 0

        epoch, instance = 0, 0

        while True:
            for batched_result in self.index_corpus.iter_word2vec(
                    counts=counts,
                    counts_sum=counts_sum,
                    sample=subsample,
                    window_size=window_size,
                    negative_samples=negative_samples,
                    negative_sampler=negative_sampler,
            ):

                (
                    positive_w_c_pair_list,
                    is_new_instance,
                    negative_c_list_as_array,
                ) = batched_result
                K = len(positive_w_c_pair_list)
                assert len(negative_c_list_as_array) == K * negative_samples
                for i, (w, c) in enumerate(positive_w_c_pair_list):
                    instance += is_new_instance[i]
                    if c != -1:
                        words[pointer] = w
                        contexts[pointer] = c
                        labels[pointer] = 1
                        pointer += 1
                        if pointer == batch_size - 1:
                            yield (([words, contexts], labels), (epoch,
                                                                 instance))
                            pointer = 0

                    for j in range(i * negative_samples,
                                   (i + 1) * negative_samples):

                        if w != -1:
                            words[pointer] = w
                            contexts[pointer] = negative_c_list_as_array[j]
                            labels[pointer] = 0
                            pointer += 1

                            if pointer == batch_size - 1:
                                yield (([words, contexts], labels), (epoch,
                                                                     instance))
                                pointer = 0

            epoch += 1
            if epochs > -1 and epoch >= epochs:
                break

    def fast_iter(self, epochs=None):
        batch_size = self.batch_size
        epochs = self.epochs if epochs is None else epochs

        subsample = self.subsample
        window_size = self.window_size
        negative_samples = self.negative_samples

        DTYPE = np.int32
        words = np.empty(shape=batch_size, dtype=DTYPE)
        contexts = np.empty(shape=batch_size, dtype=DTYPE)
        labels = np.empty(shape=batch_size, dtype=DTYPE)

        pointer = 0

        epoch, instance = 0, 0

        while True:
            for row in self.index_corpus.iter_sampled_ids(
                    counts=self.unigram_table.counts,
                    sample=subsample,
            ):
                row_length = len(row)
                for i in range(row_length):
                    window_start = max(0, i - window_size)
                    window_end = min(row_length, i + window_size + 1)
                    instance += 1

                    for j in range(window_start, window_end):
                        if i != j:
                            words[pointer] = row[j]
                            contexts[pointer] = row[i]
                            labels[pointer] = 1
                            # yield (row[j], row[i], 1), (epoch, instance)
                            pointer += 1
                            if pointer == batch_size - 1:
                                yield (([words, contexts], labels), (epoch,
                                                                     instance))
                                pointer = 0

                        for negative_id in (
                                self.unigram_table.negative_sampler.sample(
                                    negative_samples)):

                            words[pointer] = row[j]
                            contexts[pointer] = negative_samples
                            labels[pointer] = 0
                            pointer += 1
                            if pointer == batch_size - 1:
                                yield (([words, contexts], labels), (epoch,
                                                                     instance))
                                pointer = 0

                            # yield (row[j], negative_id, 0), (epoch, instance)

            epoch += 1
            if epochs > -1 and epoch >= epochs:
                break

    def iter(self, epochs=None):
        batch_size = self.batch_size
        epochs = self.epochs if epochs is None else epochs

        it = self.iter_example(epochs=epochs)

        DTYPE = np.int32
        words = np.empty(shape=batch_size, dtype=DTYPE)
        contexts = np.empty(shape=batch_size, dtype=DTYPE)
        labels = np.empty(shape=batch_size, dtype=DTYPE)

        try:
            while True:
                for i in range(batch_size):
                    (word, context, label), (epoch, instance) = next(it)
                    words[i] = word
                    contexts[i] = context
                    labels[i] = label
                yield ([words, contexts], labels), (epoch, instance)
        except StopIteration:
            pass

    def iter_example(self, epochs=None):
        subsample = self.subsample
        window_size = self.window_size
        negative_samples = self.negative_samples
        epochs = self.epochs if epochs is None else epochs

        epoch, instance = 0, 0
        while True:
            for row in self.index_corpus.iter_sampled_ids(
                    counts=self.unigram_table.counts,
                    sample=subsample,
            ):
                row_length = len(row)
                for i in range(row_length):
                    window_start = max(0, i - window_size)
                    window_end = min(row_length, i + window_size + 1)
                    instance += 1

                    for j in range(window_start, window_end):
                        if i != j:
                            yield (row[j], row[i], 1), (epoch, instance)

                        for negative_id in (
                                self.unigram_table.negative_sampler.sample(
                                    negative_samples)):
                            yield (row[j], negative_id, 0), (epoch, instance)

            epoch += 1
            if epochs > -1 and epoch >= epochs:
                break

    def logging_debug(self, emb, nb_example=40):
        logging.info('logging word2vec iterator')
        vocab = emb.get_vocab()
        for i, blob in enumerate(self.iter_example()):
            if nb_example > -1 and i >= nb_example:
                break
            (word, context, label), _ = blob
            word = vocab[word]
            context = vocab[context]
            logging.info('%20s %20s %s', word, context, '+'
                         if label == 1 else '-')

    def logging_debug_fast2_iter(self, emb, nb_example=40):
        logging.info('logging word2vec fast2 iter')
        vocab = emb.get_vocab()

        it = self.fast2_iter()
        while nb_example > 0:
            (wc, label), _ = next(it)
            word, context = wc

            for i in range(min(nb_example, len(word))):
                str_word = vocab[word[i]]
                str_context = vocab[context[i]]
                str_label = '+' if label[i] == 1 else '-'
                logging.info('%20s %20s %s', str_word, str_context, str_label)

            nb_example -= min(nb_example, len(word))


class BilbowaIterator(object):
    def __init__(
            self,
            lang0_index_corpus,
            lang1_index_corpus,
            lang0_unigram_table,
            lang1_unigram_table,
            subsample,
            length,
            batch_size,
            epochs=-1
    ):
        self.lang0_index_corpus = lang0_index_corpus
        self.lang1_index_corpus = lang1_index_corpus
        self.lang0_unigram_table = lang0_unigram_table
        self.lang1_unigram_table = lang1_unigram_table

        self.subsample = subsample
        self.length = length
        self.batch_size = batch_size
        self.epochs = epochs



    def iter(self, epochs=None):
        length = self.length
        batch_size = self.batch_size
        epochs = self.epochs if epochs is None else epochs

        it = self.iter_example(epochs=epochs)

        sent_0 = np.empty(shape=(batch_size, length), dtype=np.int32)
        sent_1 = np.empty(shape=(batch_size, length), dtype=np.int32)
        mask_0 = np.empty(shape=(batch_size, length), dtype=np.float32)
        mask_1 = np.empty(shape=(batch_size, length), dtype=np.float32)
        y = np.empty(shape=(batch_size), dtype=np.float32)  # dummy

        try:
            while True:
                sent_0[:], sent_1[:] = 0, 0
                mask_0[:], mask_1[:] = 0.0, 0.0
                y[:] = 0.0
                for i in range(batch_size):
                    (row_0, row_1), (epoch, instance) = next(it)
                    row_0, row_1 = row_0[:length], row_1[:length]
                    length_0, length_1 = len(row_0), len(row_1)
                    sent_0[i, :length_0] = row_0
                    sent_1[i, :length_1] = row_1
                    mask_0[i, :length_0] = 1.0
                    mask_1[i, :length_1] = 1.0
                    # print("sent_0", sent_0[i, :length_0])
                    # print("sent_1",sent_1[i, :length_1])
                    # print("mask_0", mask_0[i, :length_0])
                # print(sent_0)
                # print(mask_0)
                yield ([sent_0, mask_0, sent_1, mask_1], y), (epoch, instance)
        except StopIteration:
            pass

    def iter_example(self, epochs=None):
        subsample = self.subsample
        epochs = self.epochs if epochs is None else epochs

        epoch, instance = 0, 0
        while True:
            for row_0, row_1 in zip(
                    self.lang0_index_corpus.iter_sampled_ids(
                        counts=self.lang1_unigram_table.counts,
                        sample=subsample,
                    ),
                    self.lang1_index_corpus.iter_sampled_ids(
                        counts=self.lang0_unigram_table.counts,
                        sample=subsample,
                    )):

                if len(row_0) > 0 and len(row_1) > 0:
                    instance += 1
                    yield (row_0, row_1), (epoch, instance)

            epoch += 1
            if epochs > -1 and epoch >= epochs:
                break

    def logging_debug(self, emb, nb_example=500):
        logging.info('logging bilbowa iterator')
        vocab = emb.get_vocab()
        for i, blob in enumerate(self.iter_example()):
            if nb_example > -1 and i >= nb_example:
                break
            (row_0, row_1), _ = blob

            logging.info('lang 0 sent %d %s', i,
                         ' '.join([vocab[_] for _ in row_0]))
            logging.info('lang 1 sent %d %s', i,
                         ' '.join([vocab[_] for _ in row_1]))



def pair2id(strong_pairs, weak_pairs, emb):
    strong_id = set()
    weak_id = set()
    l0_dict = {}
    l1_dict = {}

    for pair in strong_pairs:
        f=emb.encode(pair[0], lang_id=1)
        e=emb.encode(pair[1], lang_id=0)
        if(f == -1 or e == -1):
            print("delete pair")
            continue
        strong_id.add((f,e))
        l1_dict.setdefault(f, set()).add(e)
        l0_dict.setdefault(e, set()).add(f)



    for pair in weak_pairs:
        f=emb.encode(pair[0], lang_id=1)
        e=emb.encode(pair[1], lang_id=0)
        if (f == -1 or e == -1):
            print("delete pair")
            continue
        weak_id.add((f,e))
        l1_dict.setdefault(f, set()).add(e)
        l0_dict.setdefault(e, set()).add(f)

    return strong_id, weak_id, l0_dict, l1_dict

def read_pair():
    strong = set()
    with open("../../data/train/strong.txt") as f:
        for line in f:
            line = line.strip().split("\t")
            strong.add((line[0],line[1]))

    weak = set()
    with open("../../data/train/weak.txt") as f:
        for line in f:
            line = line.strip().split("\t")
            weak.add((line[0], line[1]))

    return strong, weak


class strong_pairIterator(object):
    def __init__(self,
                 strong_id,
                 l0_unigram_table,
                 l1_unigram_table,
                 batch_size,
                 negative_samples,
                 l0_dict,
                 l1_dict,
                 epochs=-1
                 ):
        self.batch_size = batch_size
        self.strong_id = strong_id
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.l0_unigram_table = l0_unigram_table
        self.l1_unigram_table = l1_unigram_table
        self.l0_dict = l0_dict
        self.l1_dict = l1_dict


    def strong_iter(self, epochs=None):
        batch_size = self.batch_size
        it = self.strong_iter_example(epochs=epochs)
        epochs = self.epochs if epochs is None else epochs

        l0_s_words = np.zeros(shape=batch_size, dtype=np.int32)
        l1_s_words = np.zeros(shape=batch_size, dtype=np.int32)
        y = np.empty(shape=batch_size, dtype=np.float32)  # dummy

        try:
            while True:
                for i in range(batch_size):
                    pair, label, (epoch, instance) = next(it)
                    l0_s_words[i] = pair[0]
                    l1_s_words[i] = pair[1]
                    if(label == -1):
                        y[i] = label * 0.01
                    else:
                        y[i] = label
                    # print(l0_s_words[i], l1_s_words[i], label)

                yield ([l0_s_words, l1_s_words], y), (epoch, instance)

        except StopIteration:
            pass


    def strong_iter_example(self, epochs=None):
        epochs = self.epochs if epochs is None else epochs
        epoch, instance = 0, 0
        while True:
            for pair in self.strong_id:
                instance += 1
                yield pair, 1, (epoch, instance)

                l0_negative_list = self.l0_unigram_table.negative_sampler.sample(int(self.negative_samples/2)+3)
                l1_negative_list = self.l1_unigram_table.negative_sampler.sample(int(self.negative_samples/2)+3)
                count = 0
                for id in l0_negative_list: ##l0 = en
                    if(id in self.l1_dict[pair[0]]): ##l1 = fr
                        # print("CORRUPT")
                        continue
                    else:
                        if(count < self.negative_samples/2):
                            neg_pair = (pair[0],id)
                            count += 1
                            yield neg_pair, -1, (epoch, instance)

                for id in l1_negative_list: #fr
                    if(id in self.l0_dict[pair[1]]):
                        # print("CORRUPT")
                        continue
                    else:
                        if(count < self.negative_samples):
                            neg_pair = (id, pair[1])
                            count += 1
                            yield neg_pair, -1, (epoch, instance)

            epoch += 1
            if epochs > -1 and epoch >= epochs:
                break


class weak_pairIterator(object):
    def __init__(self,
                 weak_id,
                 l0_unigram_table,
                 l1_unigram_table,
                 batch_size,
                 negative_samples,
                 l0_dict,
                 l1_dict,
                 epochs=-1
                 ):
        self.batch_size = batch_size
        self.weak_id = weak_id
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.l0_unigram_table = l0_unigram_table
        self.l1_unigram_table = l1_unigram_table
        self.l0_dict = l0_dict
        self.l1_dict = l1_dict


    def weak_iter(self, epochs=None):
        batch_size = self.batch_size
        it = self.weak_iter_example(epochs=epochs)
        epochs = self.epochs if epochs is None else epochs

        l0_w_words = np.zeros(shape=batch_size, dtype=np.int32)
        l1_w_words = np.zeros(shape=batch_size, dtype=np.int32)
        labels = np.zeros(shape=batch_size, dtype=np.int32)
        y = np.empty(shape=batch_size, dtype=np.float32)  # dummy

        try:
            while True:
                y[:] = 0
                for i in range(batch_size):
                    pair, label, (epoch, instance) = next(it)
                    l0_w_words[i] = pair[0]
                    l1_w_words[i] = pair[1]
                    if (label == -1):
                        y[i] = label * 0.01
                    else:
                        y[i] = label

                yield ([l0_w_words, l1_w_words], y), (epoch, instance)

        except StopIteration:
            pass


    def weak_iter_example(self, epochs=None):
        epochs = self.epochs if epochs is None else epochs


        epoch, instance = 0, 0
        while True:
            for pair in self.weak_id:
                instance += 1
                yield pair,1,(epoch,instance)

                l0_negative_list = self.l0_unigram_table.negative_sampler.sample(int(self.negative_samples / 2) + 3)
                l1_negative_list = self.l1_unigram_table.negative_sampler.sample(int(self.negative_samples / 2) + 3)
                count = 0
                for id in l0_negative_list:
                    if (id in self.l1_dict[pair[0]]):
                        continue
                    else:
                        if (count < self.negative_samples / 2):
                            neg_pair = (pair[0],id)
                            count += 1
                            yield neg_pair, -1, (epoch, instance)
                        else:
                            break

                for id in l1_negative_list:
                    if (id in self.l0_dict[pair[1]]):
                        continue
                    else:
                        if (count < self.negative_samples):
                            neg_pair = (id, pair[1])
                            count += 1
                            yield neg_pair, -1, (epoch, instance)
                        else:
                            break


            epoch += 1
            if epochs > -1 and epoch >= epochs:
                break


















