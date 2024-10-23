#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Helper functions to load BeIR data from local folder
'''
import os
import csv
import orjson
from pathlib import Path

# Assume we have download & unzip the beir corpus to local folder "./beir"
BEIR_ROOT_FOLDER = os.path.join(os.path.split(os.path.realpath(__file__))[0], "beir")

BEIR_TASKS = {
    "ArguAna": "arguana",
    "ClimateFEVER": "climate-fever",
    "CQADupstackAndroidRetrieval": "cqadupstack/android",
    "CQADupstackEnglishRetrieval": "cqadupstack/english",
    "CQADupstackGamingRetrieval": "cqadupstack/gaming",
    "CQADupstackGisRetrieval": "cqadupstack/gis",
    "CQADupstackMathematicaRetrieval": "cqadupstack/mathematica",
    "CQADupstackPhysicsRetrieval": "cqadupstack/physics",
    "CQADupstackProgrammersRetrieval": "cqadupstack/programmers",
    "CQADupstackStatsRetrieval": "cqadupstack/stats",
    "CQADupstackTexRetrieval": "cqadupstack/tex",
    "CQADupstackUnixRetrieval": "cqadupstack/unix",
    "CQADupstackWebmastersRetrieval": "cqadupstack/webmasters",
    "CQADupstackWordpressRetrieval": "cqadupstack/wordpress",
    "DBPedia": "dbpedia-entity",
    "FEVER": "fever",
    "FiQA2018": "fiqa",
    "HotpotQA": "hotpotqa",
    "MSMARCO": "msmarco",
    "NFCorpus": "nfcorpus",
    "NQ": "nq",
    "QuoraRetrieval": "quora",
    "SCIDOCS": "scidocs",
    "SciFact": "scifact",
    "Touche2020": "webis-touche2020",
    "TRECCOVID": "trec-covid",
}

def load_beir_data(task_name: str, splits: list[str] = ['test']):
    local_folder = Path(BEIR_ROOT_FOLDER) / BEIR_TASKS[task_name]
    assert local_folder.exists(), f"{local_folder} does not exists. Please check your downloads."
    
    corpus_path = local_folder / "corpus.jsonl"
    queries_path = local_folder / "queries.jsonl"
    qrels_folder = local_folder / "qrels"
    
    corpus: dict[str, dict[str, str]] = dict()
    with open(corpus_path, 'r') as f:
        for line in f:
            item: dict[str, any] = orjson.loads(line)
            corpus[item["_id"]] = {
                "title": item.get("title", ""),
                "text": item["text"]
            }
    
    queries: dict[str, str] = dict()
    with open(queries_path, 'r') as f:
        for line in f:
            item: dict[str, any] = orjson.loads(line)
            queries[item["_id"]] = item["text"]
    
    qrels: dict[dict[str, dict[str, int]]] = dict()
    for _split in splits:
        qrels[_split] = dict()

        with open(qrels_folder / (_split + '.tsv'), 'r') as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            next(reader)
            
            for row in reader:
                query_id, corpus_id, score = row[0], row[1], int(row[2])
                
                if query_id not in qrels[_split]:
                    qrels[_split][query_id] = {corpus_id: score}
                else:
                    qrels[_split][query_id][corpus_id] = score

    # BeIR dataset share the same corpus across all splits
    corpus_full = {_split: corpus for _split in splits}

    # BeIR dataset stucks all queires in one file
    # Thus we need to filter out the queries appears in the qrels
    queries_full = {_split: {_id: queries[_id] for _id in qrels[_split].keys()} for _split in splits}

    return corpus_full, queries_full, qrels

if __name__ == '__main__':
    corpus, queries, qrels = load_beir_data('NFCorpus', splits=["dev", "test"])
