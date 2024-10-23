#!/bin/bash
#
# Download BEIR Datasets
#
# Links:
# 
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip
# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
# 
# [Optional] Private Datasets Download: (bioasq signal1m trec-news robust04)
#
# https://github.com/beir-cellar/beir/issues/86
#
#

DATASETS=(msmarco trec-covid nfcorpus nq hotpotqa fiqa arguana webis-touche2020 cqadupstack quora dbpedia-entity scidocs fever climate-fever scifact)

# Download zips
mkdir -p beir_zips
for DATASET_NAME in ${DATASETS[@]}; do
    echo Downloading ${DATASET_NAME}
    wget -cP beir_zips/ https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${DATASET_NAME}.zip
done

# Unzip
mkdir -p beir
for DATASET_NAME in ${DATASETS[@]}; do
    echo Unziping ${DATASET_NAME}
    unzip beir_zips/${DATASET_NAME}.zip -d beir
done