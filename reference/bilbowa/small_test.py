import numpy as np
import pickle
from data import *
from annoy import AnnoyIndex
import random


emb0 = Embedding("/Applications/Setapp/GD/research/cross-lingual/bilingual_dict_embeddings/reference/bilbowa/save_embed/withctx.en-fr.en.50.1.txt")
emb1 = Embedding("/Applications/Setapp/GD/research/cross-lingual/bilingual_dict_embeddings/reference/bilbowa/save_embed/withctx.en-fr.fr.50.1.txt")

# emb0 = Embedding("./data_root/withctx.en-fr.en.50.1.txt")
# emb1 = Embedding("./data_root/withctx.en-fr.fr.50.1.txt")


car_index = emb0.vocab2id["appetite"]
car_emb = emb0.emb[car_index]
car_emb = np.array(car_emb)
embedding0 = np.array(emb0.emb)


voiture_index = emb1.vocab2id["appétit"]
voiture_emb = emb1.emb[voiture_index]
voiture_emb = np.array(voiture_emb)
embedding1 = np.array(emb1.emb)

# nearest neighbour
product = np.dot(voiture_emb, np.transpose(embedding0))
top_k = product.argsort()[-20:][::-1]
for k in top_k:
    print(emb0.id2vocablower[k])
    print(product[k])

print("||||||||||||")
product_1 = np.dot(voiture_emb, np.transpose(car_emb))
print("product_1", product_1)


print("///////")
f = 50
t = AnnoyIndex(f, metric = "euclidean") #euclidean,angular
i = 0
for emb in embedding0:
    t.add_item(i, emb)
    i += 1

t.build(30)
top_k = t.get_nns_by_vector(voiture_emb, 20)
for k in top_k:
    print(emb0.id2vocablower[k])


