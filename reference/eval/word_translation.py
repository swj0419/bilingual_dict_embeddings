# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import io
from logging import getLogger
import numpy as np
import torch




DIC_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.', 'data', 'crosslingual', 'dictionaries')


logger = getLogger()


def load_identical_char_dico(word2id1, word2id2):
    """
    Build a dictionary of identical character strings.
    """
    pairs = [(w1, w1) for w1 in word2id1.keys() if w1 in word2id2]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings." % len(pairs))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def load_dictionary(path, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, dico_eval):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
    if dico_eval == 'default':
        path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
    else:
        path = dico_eval
    dico = load_dictionary(path, word2id1, word2id2)
    # dico = dico.cuda() if emb1.is_cuda else dico

    # assert dico[:, 0].max() < emb1.size(0)
    # assert dico[:, 1].max() < emb2.size(0)

    # # normalize word embeddings
    # emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    # emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors

    query = emb1[dico[:, 0]]
    scores = query.mm(emb2.transpose(0, 1))

    results = []
    top_matches = scores.topk(10, 1, True)[1]
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        print("precision", precision_at_k)
        logger.info("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    return results
