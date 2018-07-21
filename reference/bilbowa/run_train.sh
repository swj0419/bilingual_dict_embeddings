#!/usr/bin/env bash


data_root=~/rstore/repos/bilingual_dict_embeddings/reference/bilbowa/data_root
model_root=~/rstore/repos/bilingual_dict_embeddings/reference/bilbowa/model_root

mkdir -p "$data_root"
mkdir -p "$model_root"

data_root=`realpath "$data_root"`
model_root=`realpath "$model_root"`

./train.py \
  --data_root "$data_root" \
  --lang0_emb_file withctx.en-fr.en.50.1.txt \
  --lang1_emb_file withctx.en-fr.fr.50.1.txt \
  --lang0_ctxemb_file withctx.en-fr.en.50.1.txt.ctx \
  --lang1_ctxemb_file withctx.en-fr.fr.50.1.txt.ctx \
  --mono_max_lines 10000 \
  --multi_max_lines 10000 \
  --model_root "$model_root" \
  --lang0_mono_index_corpus_file en_mono \
  --lang1_mono_index_corpus_file fr_mono \
  --lang0_multi_index_corpus_file en_multi \
  --lang1_multi_index_corpus_file fr_multi \
  --emb_dim 50 \
  --word2vec_batch_size 100000 \
  --bilbowa_sent_length 50 \
  --bilbowa_batch_size 100 \
  --train_mono=true \
  --train_multi=true \
  ;
