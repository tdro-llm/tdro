#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Helper functions to load MIRACL data from local folder

'''
import os
import gzip
import orjson

# Assume we have clone the miracl & miracl-corpus corpus to current local folder
MIRACL_ROOT_FOLDER = os.path.join(os.path.split(os.path.realpath(__file__))[0], "miracl")
MIRACL_CORPUS_ROOT_FOLDER = os.path.join(os.path.split(os.path.realpath(__file__))[0], "miracl-corpus")

_EVAL_SPLIT = "dev"
_LANGS = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh', 'de', 'yo']

languages2filesize = { 
    'ar': 5,
    'bn': 1,
    'en': 66 ,
    'es': 21,
    'fa': 5,
    'fi': 4,
    'fr': 30,
    'hi': 2,
    'id': 3,
    'ja': 14,
    'ko': 3,
    'ru': 20,
    'sw': 1,
    'te': 2,
    'th': 2,
    'zh': 10,
    'de': 32,
    'yo': 1,
}

DATASET_PATHS = {
    lang: {
        'train': [
            f'{MIRACL_CORPUS_ROOT_FOLDER}/miracl-corpus-v1.0-{lang}/docs-{i}.jsonl.gz' for i in range(n)
        ]
    } for lang, n in languages2filesize.items()
}

def load_miracl_data(langs: list[str], splits: list[str], path: str = MIRACL_ROOT_FOLDER):
    queries = {}     # queries[lang][split][query_id] = query_text
    corpus = {}      # corpus[lang][split][docid] = {"text": text}
    relevant_docs = {}   # relevant_docs[lang][split][query_id][docid] = 1

    for lang in langs:
        queries[lang], corpus[lang], relevant_docs[lang] = {}, {}, {}

        # All splits share the same corpus of this lang
        # Thus only need to load once
        corpus_lang = {}
        for filepath in DATASET_PATHS[lang]['train']:
            with gzip.open(filepath) as f:
                for line in f:
                    item: dict[str, any] = orjson.loads(line)
                    corpus_lang[item["docid"]] = {"title": item["title"], "text": item["text"]}
        
        for split in splits:
            queries[lang][split], corpus[lang][split], relevant_docs[lang][split] = {}, {}, {}

            # Let's assume `path` is `xxx/miracl`, and the corpus is located in `xxx/miracl-corpus`
            # Load Query
            with open(os.path.join(path, f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{split}.tsv")) as f:
                for line in f:
                    query_id, query_text = line.split('\t')
                    queries[lang][split][query_id] = query_text
            
            # Assign loaded corpus
            corpus[lang][split] = corpus_lang
            
            # Load Qrels
            with open(os.path.join(path, f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv")) as f:
                for line in f:
                    query_id, Q0, docid, relevance = line.split('\t')
                    if int(relevance) >= 1:
                        if query_id not in relevant_docs[lang][split]:
                            relevant_docs[lang][split][query_id] = {}
                        relevant_docs[lang][split][query_id][docid] = 1
    
    return corpus, queries, relevant_docs


if __name__ == '__main__':
    corpus, queries, relevant_docs = load_miracl_data(langs=['ar', 'bn'], splits=["dev"])
    print()