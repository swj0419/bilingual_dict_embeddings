import pickle


import math
import os
from os.path import join
import time
import sys
import random

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import numpy as np

from keras.optimizers import Adam
import tensorflow as tf
from reference.bilbowa.data import *



def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# store vocab2id
emb0 = Embedding("/Applications/Setapp/GD/research/cross-lingual/bilingual_dict_embeddings/reference/bilbowa/data_root/small_withctx.en-fr.en.50.1.txt")
emb1 = Embedding("/Applications/Setapp/GD/research/cross-lingual/bilingual_dict_embeddings/reference/bilbowa/data_root/small_withctx.en-fr.fr.50.1.txt")


# write dictionary
'''
emb = MultiLanguageEmbedding(emb0, emb1)
vocab = emb.get_vocab()

emb0_vocab = emb0.id2vocablower
emb1_vocab = emb1.id2vocablower

with open('emb0_id2vocab.pickle', 'wb') as handle:
    pickle.dump(emb0_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('emb1_id2vocab.pickle', 'wb') as handle:
    pickle.dump(emb1_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''

# read original id2vocab
with open('./data_root/emb0_id2vocab.pickle', 'rb') as handle:
    prev_emb0_id2vocab = pickle.load(handle)

with open('./data_root/emb1_id2vocab.pickle', 'rb') as handle:
    prev_emb1_id2vocab = pickle.load(handle)

'''
# store the embedding only appears in the strong weak pair
English = set()
French = set()

with open("../../data/train/strong.txt") as f:
    for line in f:
        line = line.strip().split("\t")
        French.add(line[0])
        English.add(line[1])

with open("../../data/train/weak.txt") as f:
    for line in f:
        line = line.strip().split("\t")
        French.add(line[0])
        English.add(line[1])



# stop words
en_stop = {}
with open("./data_root/stop_words_en.txt") as f:
    for line in f:
        line = line.strip().lower()
        en_stop[line] = 1

fr_stop = {}
with open("./data_root/stop_words_fr.txt") as f:
    for line in f:
        line = line.strip().lower()
        fr_stop[line] = 1


# English
for x in range(50000-26101):
    word = prev_emb0_id2vocab[x]
    if(hasNumbers(word)):
        continue
    if(word in en_stop):
        continue
    English.add(word)

for x in range(50000-24730):
    word = prev_emb1_id2vocab[x]
    if (hasNumbers(word)):
        continue
    if (word in fr_stop):
        continue
    French.add(word)

print("English", len(English))
print("French", len(French))
'''

# mono file
with open("./data_root/en_multi.ids.txt", "r") as f:
    with open("./data_root/small_en_multi.ids.txt", "w") as f1:
        for line in f:
            lines = line.strip().split()
            for w in lines:
                word = prev_emb0_id2vocab[int(w)]
                print(word)
                if word in emb0.vocab:
                    f1.write(str(emb0.vocab2id[word]))
                    f1.write(' ')
            f1.write('\n')

print("done")

with open("./data_root/fr_multi.ids.txt", "r") as f:
    with open("./data_root/small_fr_multi.ids.txt", "w") as f1:
        for line in f:
            lines = line.strip().split()
            for w in lines:
                word = prev_emb1_id2vocab[int(w)-995000]
                if word in emb1.vocab:
                    f1.write(str(emb1.vocab2id[word]+39016))
                    f1.write(' ')
            f1.write('\n')



#
# with open("./data_root/withctx.en-fr.en.50.1.txt.ctx","r") as f:
#     with open("./data_root/small_withctx.en-fr.en.50.1.txt.ctx", "w") as f1:
#         for line in f:
#             lines = line.strip().split()
#             if (lines[0] in English):
#                 f1.writelines(line)
#
#
# with open("./data_root/withctx.en-fr.fr.50.1.txt.ctx","r", encoding = "ISO-8859-1") as f:
#     with open("./data_root/small_withctx.en-fr.fr.50.1.txt.ctx", "w") as f1:
#         for line in f:
#             lines = line.strip().split()
#             if (lines[0] in French):
#                 f1.writelines(line)
