import os
import orjson
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Union
from itertools import chain

import jieba
import jionlp as jio
from simhash import Simhash, SimhashIndex

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

os.chdir(os.path.split(os.path.realpath(__file__))[0])


def format_query(query: str):
    formated_query = query.replace("_", "")     # Remove `_`
    if jio.check_any_chinese_char(formated_query):
        # Chinese characters, do split manually
        formated_query = " ".join(jieba.lcut(formated_query))
    return formated_query


def load_beir_test_sets(
    query_collection_folder: str = "PATH_TO/ALL_test_queries"
):
    texts: List[Dict[str, any]] = []
    for input_file in tqdm(Path(query_collection_folder).iterdir(), desc="Loading BEIR Test Queries"):
        if input_file.suffix not in [".json", ".jsonl"]:
            continue
        with open(input_file, "r") as f:
            for line in f:
                item = orjson.loads(line)
                item["query"] = format_query(item["query"])
                if not item["query"]:
                    continue
                item["meta"] = {"filename": input_file.name}
                texts.append(item)
    return texts

def load_miracl_dev_sets(
    path: str = "PATH_TO/miracl",
    langs: List[str] = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh', 'de', 'yo'],
):
    texts: List[Dict[str, any]] = []
    for lang in tqdm(langs, desc="Loading MIRACL Test Queries"):
        # Let's assume `path` is `xxx/miracl`, and the corpus is located in `xxx/miracl-corpus`
        # Load Query
        input_file = Path(path) / f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv"
        with open(input_file, "r") as f:
            for line in f:
                query_id, query_text = line.split('\t')
                query_text = format_query(query_text)
                if not query_text:
                    continue
                item = {
                    "query_id": query_id,
                    "query": query_text,
                    "meta": {"filename": input_file.name}
                }
                texts.append(item)
    return texts


def main(
        input_files: List[str],
        dedup_output_folder = "retrieval/dedup",
        duplicates_output_folder = "retrieval/filtered_dups",
    ):
    dedup_output_folder = Path(dedup_output_folder)
    duplicates_output_folder = Path(duplicates_output_folder)
    dedup_output_folder.mkdir(parents=True, exist_ok=True)
    duplicates_output_folder.mkdir(parents=True, exist_ok=True)

    # Load Datasets
    beir_texts = load_beir_test_sets()
    miracl_texts = load_miracl_dev_sets()

    texts: Dict[str, Dict[str, any]] = {str(idx): item for idx, item in enumerate(chain(beir_texts, miracl_texts))}
    logger.info(f"{len(texts)} samples loaded.")

    # Build Simhash objs
    objs = []
    for idx, item in tqdm(texts.items(), desc="Building Simhash objs"):
        objs.append((idx, Simhash(item["query"])))

    index = SimhashIndex(objs, k=2)

    # Get queries
    for filepath in input_files:
        filepath = Path(filepath)

        dedup_filepath = dedup_output_folder / filepath.name
        dedup_fp = open(dedup_filepath, "w")

        dups_filepath = duplicates_output_folder / filepath.name
        dup_items = []

        with open(filepath, "r") as f:
            for line in tqdm(f, desc=f"Inspecting Dups {filepath.name}", mininterval=1000):
                item = orjson.loads(line)
                query = format_query(item["query"])
                query_obj = Simhash(query)
                dups = index.get_near_dups(query_obj)
                if len(dups) > 0:
                    dup_item = {"query_id": item["query_id"], "query": item["query"], "test_dups": []}
                    logger.info(f"\n=== Finding Dups === \nQuery: \n{item['query']}\nTraining Dups: ")
                    for dup_idx in dups:
                        logger.info(texts[dup_idx])
                        dup_item["test_dups"].append(texts[dup_idx])
                    dup_items.append(dup_item)
                else:
                    dedup_fp.write(line)
        
        if dup_items:
            with open(dups_filepath, "wb") as f:
                f.write(orjson.dumps(dup_items, option=orjson.OPT_INDENT_2)) 

        dedup_fp.close()



if __name__ == '__main__':
    domain_names = [
        'agnews',
        'AllNLI',
        'altlex',
        'amazon_review_2018_1m',
        'cnn_dailymail',
        'codesearchnet',
        'dureader',
        'eli5_question_answer',
        'gooaq_pairs',
        'hotpotqa',
        'medmcqa',
        'miracl',
        'mr_tydi_combined',
        'msmarco',
        'nq',
        'quora_duplicates_triplets',
        'searchQA_top5_snippets',
        'sentence-compression',
        'SimpleWiki',
        'squad_pairs',
        'stackexchange_duplicate_questions_title-body_title-body',
        't2ranking',
        'trivia',
        'xsum',
        'yahoo_answers_title_answer',
    ]

    input_files = [f"retrieval/original/{i}.jsonl" for i in domain_names]
    main(input_files=input_files)
