import pickle
import numpy as np
import sys
sys.path.insert(0, '../eval')
from evaluate import Evaluator
from absl import logging
import torch


emb0 = pickle.load(open("./sav_model/0828_dim50_s0_0.001_Pretrained/emb0.pickle", "rb", -1))
emb1 = pickle.load(open("./sav_model/0828_dim50_s0_0.001_Pretrained/emb1.pickle", "rb", -1))

emb0_word2id_200k = pickle.load(open("./data_root/emb0_word2id_200k.pickle", "rb", -1))
emb1_word2id_200k = pickle.load(open("./data_root/emb1_word2id_200k.pickle", "rb", -1))

emb0_200k = np.empty((len(emb0_word2id_200k),50), dtype = "f")
count = 0
emb0_word2id_200k_new = emb0_word2id_200k.copy()
for word,id in emb0_word2id_200k.items():
    if(word in emb0.vocablower2id):
        word_emb = emb0.emb[emb0.vocab2id[word]]
        emb0_200k[id, :] = word_emb
    else:
        del(emb0_word2id_200k_new[word])
        emb0_200k[id, :] = np.random.uniform(low=100, high=1000, size=(1,50))
        count += 1
print(count)

emb1_200k = np.empty((len(emb1_word2id_200k),50), dtype = "f")
count = 0
emb1_word2id_200k_new = emb1_word2id_200k.copy()
for word,id in emb1_word2id_200k.items():
    if(word in emb1.vocablower2id):
        # print(word, emb1.id2vocablower[emb1.vocablower2id[word]])
        word_emb = emb1.emb[emb1.vocab2id[word]]
        emb1_200k[id, :] = word_emb
    else:
        del (emb1_word2id_200k_new[word])
        emb1_200k[id, :] = np.random.uniform(low=100, high=1000, size=(1,50))
        count += 1
        # print(".....")

print(count)

#################

# voiture_index = emb1_word2id_200k_new["voiture"]
# voiture_emb =emb1_200k[voiture_index]
# car_index = emb0_word2id_200k_new["car"] # disappear, appetite
# car_emb = emb0_200k[car_index]
#
#
#
# embedding0 =  torch.tensor(emb0_200k)
# embedding0 = embedding0 / embedding0.norm(2, 1, keepdim=True).expand_as(embedding0)
# embedding1 =  torch.tensor(emb1_200k)
# embedding1 = embedding1 / embedding1.norm(2, 1, keepdim=True).expand_as(embedding1)
# voiture_emb = embedding1[voiture_index]
# voiture_emb = np.array(voiture_emb)
# car_emb = embedding0[car_index]
# car_emb = np.array(car_emb)
# product_1 = np.dot(voiture_emb, np.transpose(car_emb))
# print("product_1", product_1)
#
# product = np.dot(np.array(voiture_emb), np.transpose(np.array(embedding0)))
# print(product)
# top_k = product.argsort()[-20:][::-1]
# # print(emb0_word2id_200k_new[859])
# for k in top_k:
#     if(k not in emb0_word2id_200k_new):
#         print(k)
#         print(emb0_200k[k])
#         print(product[k])
#         print("......")
#     else:
#         print(emb0_word2id_200k_new[k])
#         print(product[k])
#         print("///")
#







print("en-fr_test")
evaluator = Evaluator(emb0_200k, emb1_200k, emb0_word2id_200k_new, emb1_word2id_200k_new, "en", "fr", "default")
results = evaluator.word_translation()[0]
print("----------------------------------------------------------------------------------")
print("en-fr_test")
evaluator = Evaluator(emb0.emb,emb1.emb, emb0.vocablower2id, emb1.vocablower2id, "en", "fr", "default")
results = evaluator.word_translation()[0]


# fr - en
print("fr-en_test")
evaluator = Evaluator(emb1_200k, emb0_200k, emb1_word2id_200k_new, emb0_word2id_200k_new, "fr", "en", "default")
results = evaluator.word_translation()[0]
print("----------------------------------------------------------------------------------")
print("fr-en_test")
evaluator = Evaluator(emb1.emb,emb0.emb, emb1.vocablower2id,emb0.vocablower2id, "fr", "en", "default")
results = evaluator.word_translation()[0]


