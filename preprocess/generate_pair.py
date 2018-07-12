import csv
import numpy as np
from gensim.models import KeyedVectors




### store stop word
en_stop = {}
with open("../data/stop/stop_words_en.txt") as f:
    for line in f:
        line = line.strip().lower()
        en_stop[line] = 1

fr_stop = {}
with open("../data/stop/stop_words_fr.txt") as f:
    for line in f:
        line = line.strip().lower()
        fr_stop[line] = 1

es_stop = {}
with open("../data/stop/stop_words_es.txt") as f:
    for line in f:
        line = line.strip().lower()
        es_stop[line] = 1


#####for Dictionary
## dict variables
exact_fr_en = set()
en_fr_pair = {}
fr_en_pair = {}
fr_unique = set()
en_unique = set()

## read en-fr dict
spamReader = csv.reader(open('../data/train/en_fr_train500_4.csv'), delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
for row in spamReader:
    ##find exact pair
    exact = (row[0], row[1])
    exact_fr_en.add(exact)

    ##unique
    en_unique.add(row[1])
    fr_unique.add(row[0])

    ##find pairs
    key = row[1]
    definition = row[2].split()
    value = set()

    ##delete stop word
    for i in definition:
        if i in fr_stop:
            pass
        else:
            value.add(i.lower())
            fr_unique.add(i.lower())

    en_fr_pair.update({key: value})



## read fr-en dict
spamReader = csv.reader(open('../data/train/fr_en_train500_4.csv'), delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
for row in spamReader:
    ##find exact pair
    exact = (row[1], row[0])
    exact_fr_en.add(exact)

    ##unique
    en_unique.add(row[0])
    fr_unique.add(row[1])


    ##find pairs
    key = row[0]
    definition = row[2].split()
    value = set()

    ##delete stop word
    for i in definition:
        if i in en_stop:
            pass
        else:
            value.add(i.lower())
            en_unique.add(i.lower())


    fr_en_pair.update({key: value})

strong = set()



####find pair through embeddings:
en_url = "../data/mono_embedding/wiki.en.vec"
fr_url = "../data/mono_embedding/wiki.fr.vec"
print("load embedding")
en_model = KeyedVectors.load_word2vec_format(en_url)
print("load embedding")
fr_model = KeyedVectors.load_word2vec_format(fr_url)


K = 3
print("find similar words")
# Finding out similar words [default= top 10]
for fr, en in exact_fr_en:
    find_similar_to = en
    print(find_similar_to)
    for similar_word in en_model.similar_by_word(find_similar_to,K):
        strong.add((fr,similar_word))
        print(similar_word)

    print("1")
    find_similar_to = fr
    for similar_word in fr_model.similar_by_word(find_similar_to,K):
        strong.add((similar_word,en))


print(len(strong))

M = 0.7
for fr, en in exact_fr_en:
    ##en - fr - en def
    if fr in fr_en_pair:
        en_def = fr_en_pair[fr]
        for en_word in en_def:
            try:
                score = en_model.similarity(en,en_word)
                print(score)
            except KeyError:
                continue
            if(score > M):
                print((fr,en_word))
                strong.add((fr,en_word))

    if en in en_fr_pair:
        fr_def = en_fr_pair[en]
        for fr_word in fr_def:
            try:
                score = fr_model.similarity(fr,fr_word)
                print(score,"------")
            except KeyError:
                continue
            if(score > M):
                print((fr_word,en))
                strong.add((fr_word,en))

print(len(strong), "check2")




## find strong pair
for en_key, fr_value in en_fr_pair.items():
    ## strict strong pair def
    for fr_word in fr_value:
        if fr_word in fr_en_pair:
            en_value = fr_en_pair[fr_word]
            if en_key in en_value:
                strong.add((fr_word,en_key))

    ### generate strong pair from embedding





for fr_key, en_value in fr_en_pair.items():
    for en_word in value:
        if en_word in en_fr_pair:
            fr_value = en_fr_pair[en_word]
            if fr_key in fr_value:
                strong.add((fr_key,en_word))



'''''
#######Facebook Dataset:
txt = np.loadtxt("../data/train/en-fr.0-5000.txt", dtype=np.str)
for row in txt:
    ##find exact pair
    exact = (row[1], row[0])
    exact_fr_en.add(exact)

    ##unique
    en_unique.add(row[0])
    fr_unique.add(row[1])


txt = np.loadtxt("../data/train/fr-en.0-5000.txt", dtype=np.str)
for row in txt:
    ##find exact pair
    exact = (row[0], row[1])
    exact_fr_en.add(exact)

    ##unique:
    en_unique.add(row[1])
    fr_unique.add(row[0])


# print(exact_fr_en)
# print(strong)


#strong = strong.union(exact_fr_en)



######Fetch the definition
with open("fr_words.txt", "w") as french:
    with open("en_words.txt", "w") as english:
        for i in exact_fr_en:
            french.write(i[0]+'\n')
            english.write(i[1]+'\n')

'''


print("end")









