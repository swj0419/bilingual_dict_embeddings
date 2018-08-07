import pickle
import numpy as np
from data import *



def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# store vocab2id
emb0 = Embedding("/Applications/Setapp/GD/research/cross-lingual/bilingual_dict_embeddings/reference/bilbowa/small_data_root/withctx.en-fr.en.50.1.txt")
emb1 = Embedding("/Applications/Setapp/GD/research/cross-lingual/bilingual_dict_embeddings/reference/bilbowa/small_data_root/withctx.en-fr.fr.50.1.txt")


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

for x in range(60000-24730):
    word = prev_emb1_id2vocab[x]
    if (hasNumbers(word)):
        continue
    if (word in fr_stop):
        continue
    French.add(word)

print("English", len(English))
print("French", len(French))


## multi_ids
# with open("./data_root/en_multi.ids.txt", "r") as f:
#     with open("./data_root/small_en_multi.ids.txt", "w") as f1:
#         for line in f:
#             lines = line.strip().split()
#             for w in lines:
#                 word = prev_emb0_id2vocab[int(w)]
#                 if word in English:
#                     f1.write(str(emb0.vocab2id[word]))
#                     f1.write(' ')
#             f1.write('\n')
#
# French = set(emb1.vocab)
# with open("./data_root/fr_multi.ids.txt", "r") as f:
#     with open("./data_root/small_fr_multi.ids.txt", "w") as f1:
#         for line in f:
#             lines = line.strip().split()
#             for w in lines:
#                 word = prev_emb1_id2vocab[int(w)-995000]
#                 if word in French:
#                     f1.write(str(emb1.vocab2id[word]+39016))
#                     f1.write(' ')

# # mono ids
# French = set(emb1.vocab)
# with open("./data_root/fr_mono.ids.txt", "r") as f:
#     with open("./small_data_root/fr_mono.ids.txt", "w") as f1:
#         for line in f:
#             lines = line.strip().split()
#             for w in lines:
#                 word = prev_emb1_id2vocab[int(w)-995000]
#                 if word in French:
#                     f1.write(str(emb1.vocab2id[word]+39016))
#                     f1.write(' ')
#             f1.write('\n')
# #
# with open("./data_root/en_mono.ids.txt", "r") as f:
#     with open("./small_data_root/en_mono.ids.txt", "w") as f1:
#         for line in f:
#             lines = line.strip().split()
#             for w in lines:
#                 word = prev_emb0_id2vocab[int(w)]
#                 if word in English:
#                     f1.write(str(emb0.vocab2id[word]))
#                     f1.write(' ')
#             f1.write('\n')

# # write embedding file
# count = 0
# with open("./data_root/withctx.en-fr.fr.50.1.txt","r", errors='surrogateescape') as f:
#     with open("./data_root/small_withctx.en-fr.fr.50.1.txt", "w") as f1:
#         for line in f:
#             lines = line.strip().split()
#             if (lines[0] in French):
#                 count+=1
#                 f1.writelines(line)
#
# print(count)

# with open("./data_root/withctx.en-fr.en.50.1.txt.ctx","r") as f:
#     with open("./data_root/small_withctx.en-fr.en.50.1.txt.ctx", "w") as f1:
#         for line in f:
#             lines = line.strip().split()
#             if (lines[0] in English):
#                 f1.writelines(line)

#
# with open("./data_root/withctx.en-fr.fr.50.1.txt.ctx","r", errors='surrogateescape') as f:
#     with open("./data_root/small_withctx.en-fr.fr.50.1.txt.ctx", "w") as f1:
#         for line in f:
#             lines = line.strip().split()
#             if (lines[0] in French):
#                 f1.writelines(line)

# counts
# counts_0 = np.zeros(39016)
# with open("./small_data_root/en_mono.ids.txt", "r") as f:
#     for line in f:
#         lines = line.strip().split()
#         for w in lines:
#             counts_0[int(w)] += 1
#
# np.savez('./small_data_root/en_mono.counts.npz', counts = counts_0)


counts_1 = np.zeros(48631)
with open("./small_data_root/fr_mono.ids.txt", "r") as f:
    for line in f:
        lines = line.strip().split()
        for w in lines:
            counts_1[int(w)-39016] += 1
np.savez('./small_data_root/fr_mono.counts.npz', counts=counts_1)

