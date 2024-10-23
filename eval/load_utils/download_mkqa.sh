#!/bin/bash
#
# Download MKQA Datasets
# 
# Reference: https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB/MKQA#mkqa
#
# Links:
# 
#  - Test Queries: https://huggingface.co/datasets/Shitao/bge-m3-data/resolve/main/MKQA_test-data.zip
#  - Processed English Corpus: https://huggingface.co/datasets/BeIR/nq/blob/main/corpus.jsonl.gz
# 

mkdir -p mkqa/test; cd mkqa

# Download test queries
wget -c https://huggingface.co/datasets/Shitao/bge-m3-data/resolve/main/MKQA_test-data.zip
unzip MKQA_test-data.zip -d test/

# Download corpus
wget -c https://huggingface.co/datasets/BeIR/nq/blob/main/corpus.jsonl.gz
gzip -d corpus.jsonl.gz
mv corpus.jsonl nq.jsonl
