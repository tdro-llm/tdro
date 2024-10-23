#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Evaluate QA datasets with DRES/Reranker models. This python scripts integrates `qa/test_dpr_*.sh`
with simplified and optimized pipelines.

@Time    :   2024/06/05 17:45:47
@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import sys
import time
import gzip
import orjson
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from functools import partial
from multiprocessing.pool import Pool

import logging
logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if (int(os.getenv("LOCAL_RANK", -1)) in [0, -1]) else logging.WARN,
    force=True,
)
logger = logging.getLogger(__name__)


# Avoid following error:
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# PeftModel.from_pretreined will occupy 600MiB * 8 processes GPU memory on Rank 0 only
# if `CUDA_VISIBLE_DEVICES` is not set. However, AutoModel.from_pretrained won't. 
# I don't know why this happens. 
if local_rank := os.environ.get("LOCAL_RANK", None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)


import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch import Tensor

from transformers import HfArgumentParser, set_seed

from eval_arguments import EvalArguments
from modeling_utils import ExactSearchModel, get_mteb_prompt
from qa_utils import SimpleTokenizer, has_answers, _normalize

# Hacking with Fused Ops
from tdro.utils.monkey_patch import hacking_fused_rms_norm
hacking_fused_rms_norm()

from tdro.retriever.faiss_search import FlatIPFaissSearch

ALL_LANGS = ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it', 'ja', 'km', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh_cn', 'zh_hk', 'zh_tw']

COLLECTION_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "load_utils/mkqa")
CORPUS_FILENAME = "nq.jsonl"
# CORPUS_FILENAME = "nq.100k.debug.jsonl"

def load_mkqa_data(collection_path: str, langs: List[str]):
    """
    Read cross-lingual MKQA data. Corpus is English format, Queries are cross-lingual.
    Inputs:
     - collection_path: Path to MKQA base folder.
        `collection_path/nq.jsonl`: Tevatron format corpus.
        `collection_path/test/{lang}.jsonl`: Tevatron format query & answers.
     - langs: Languages to Load
    
    Returns:
     - query_collections: Dict[str, Dict[str, str]] = {
            lang: {"query_id": query},
            ...
        }
     - corpus: Dict[str, Dict[str, str]] = {
            "docid": {"title": title, "text": text}, 
            ...
        }
     - answers_collections: Dict[str, Dict[str, List[str]]] = {
            lang: {"query_id": ["ans1", "ans2", ...]}
        }
    """
    collection_path = Path(collection_path)
    # Read corpus
    corpus: Dict[str, Dict[str, str]] = dict()
    with open(collection_path / CORPUS_FILENAME, 'r') as f:
        for line in f:
            item: Dict[str, any] = orjson.loads(line)
            if "_id" in item:
                pid: str = item.pop("_id")
            elif "docid" in item:
                pid: str = item.pop("docid")
            else:
                raise KeyError("Please check docid key name.")
            corpus[pid] = item
    
    # Read query collections
    query_collections: Dict[str, Dict[str, str]] = dict()
    answers_collections: Dict[str, Dict[str, List[str]]] = dict()
    for lang in langs:
        query_collections[lang] = dict()
        answers_collections[lang] = dict()
        with open(collection_path / f"test/{lang}.jsonl", 'r') as f:
            for line in f:
                item: Dict[str, any] = orjson.loads(line)
                qid: str = item["id"]
                query_collections[lang][qid] = item["question"]
                answers_collections[lang][qid] = item["answers"]
    
    return query_collections, corpus, answers_collections

@dataclass
class AnnotatePair:
    qid: str
    answers: List[str]
    pid: str
    score: float
    text: Optional[str] = None
    text_tokenized: Optional[List[str]] = None
    is_revelent: int = 0

def iter_corpus(corpus: dict):
    for k, v in corpus.items():
        yield {"id": k, **v}

def tokenize_corpus(item: Dict[str, str], tokenizer: SimpleTokenizer):
    return {"id": item["id"], "text_tokenized": tokenizer.tokenize(_normalize(item["title"] + " " + item["text"])).words(uncased=True)}

def iter_rerank_scores(rerank_results: Dict[str, Dict[str, float]], answers_collection: Dict[str, List[str]], corpus=None, corpus_tokenized=None):
    for qid, _results in rerank_results.items():
        for pid, _score in _results.items():
            if corpus_tokenized is None:
                psg_text: str = corpus[pid]["title"] + " " + corpus[pid]["text"] if "title" in corpus[pid] else corpus[pid]["text"]
                yield AnnotatePair(qid=qid, answers=answers_collection[qid], pid=pid, score=float(_score), text=psg_text)
            else:
                yield AnnotatePair(qid=qid, answers=answers_collection[qid], pid=pid, score=float(_score), text_tokenized=corpus_tokenized[pid])

def iter_retrieval_scores(results: Dict[str, Dict[str, float]], answers_collection: Dict[str, List[str]], corpus=None, corpus_tokenized=None):
    for (qid, q_results_dict) in results.items():
        for (pid, _score) in q_results_dict.items():
            if corpus_tokenized is None:
                psg_text: str = corpus[pid]["title"] + " " + corpus[pid]["text"] if "title" in corpus[pid] else corpus[pid]["text"]
                yield AnnotatePair(qid=qid, answers=answers_collection[qid], pid=pid, score=float(_score), text=psg_text)
            else:
                yield AnnotatePair(qid=qid, answers=answers_collection[qid], pid=pid, score=float(_score), text_tokenized=corpus_tokenized[pid])

def annotate_func(item: AnnotatePair, tokenizer: SimpleTokenizer, regex=False):
    """ 1: revelent / 0: irrevelent """
    item.is_revelent = 1 if has_answers(answers=item.answers, tokenizer=tokenizer, text=item.text, text_tokenized=item.text_tokenized, regex=regex) else 0
    return item

# Copied from beir/retrieval/custom_metrics.py
def top_k_accuracy(
        qrels: Dict[str, Dict[str, int]], 
        results: Dict[str, Dict[str, float]], 
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    top_k_acc = {}
    
    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    logging.info("\n")
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = [item[0] for item in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]]
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"]/len(qrels), 5)
        logging.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    return top_k_acc


def main(
    args: EvalArguments, 
    model: ExactSearchModel,
    k_values: List[int]=[1, 3, 5, 10, 20, 100]
):
    """
    Test cross-lingual MKQA
    """
    task_name: str = args.task_name or "MKQA"
    task_type: str = args.task_type or "Retrieval"
    split: str = "test"
    topk = max(k_values)
    corpus: Optional[Dict[str, Dict[str, str]]] = None
    corpus_tokenized: Optional[Dict[str, Dict[str, str]]] = None # Pre-tokenized corpus, used by annotation function
    is_indexed: bool = False
    retriever: Optional[FlatIPFaissSearch] = None
    pool: Optional[Pool] = None
    annotate_tokenizer = SimpleTokenizer()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    score_path = Path(args.output_dir) / f"{task_name}{task_type}.json"
    if score_path.exists():
        with open(score_path, 'r') as f:
            all_scores: Dict[str, any] = orjson.loads(f.read())
    else:
        all_scores: Dict[str, any] = {split: {}, "evaluation_time": 0.0}

    init_time = time.time()
    args.lang = args.lang or ALL_LANGS
    for lang in args.lang:
        if (not args.overwrite_results) and (lang in all_scores[split]):
            logger.info(f"MKQA {lang} result exists, skip.")
            continue

        logger.info(f"\nTask: {task_name}{task_type}, split: {split}, language: {lang}. Running...")

        # Load retriever
        if retriever is None:
            # Customized Retriever
            retriever = FlatIPFaissSearch(model, args.batch_size, use_multiple_gpu=True)

            model.query_prompt, model.corpus_prompt = "", ""
            if args.add_prompt:
                model.query_prompt, model.corpus_prompt = get_mteb_prompt(task_name=task_name, task_type=task_type, prompt_type=args.prompt_type)
                
                logger.info('Set query prompt: \n{}\n\ncorpus prompt: \n{}\n'.format(model.query_prompt, model.corpus_prompt))

        # Load Data
        # Note: cross-lingual MKQA only uses english corpus
        if corpus is None:
            query_collections, corpus, answers_collections = load_mkqa_data(collection_path=COLLECTION_PATH, langs=args.lang)

        # ** Dense Retrieval **
        # Encode Query (Queries are encoded per languages)
        query: Dict[str, str] = query_collections[lang]
        query_ids: List[str] = list(query.keys())

        logger.info("Encoding Queries...")
        start_time = time.time()
        query_emb: Tensor = retriever.encode_queries(list(query.values()), batch_size=args.batch_size)
        logger.info(f"Query embeddings encoded in {time.time()-start_time}s.")

        # Encode Corpus (Corpus is always en. Multilingual query -> En corpus)
        if not is_indexed:
            logger.info("Sorting Corpus by document length (Longest first)...")

            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "")) + len(corpus[k].get("text", "")), reverse=True)
            corpus_list: List[Dict[str, str]] = [corpus[cid] for cid in corpus_ids]

            logger.info("Encoding Corpus in batches... Warning: This might take a while!")
            logger.info("Scoring Function: {}".format(args.score_function))
            start_time = time.time()
            corpus_emb = retriever.encode_corpus(corpus_list, batch_size=args.batch_size)
            logger.info(f"Corpus embeddings encoded in {time.time()-start_time}s.")

            # Index
            retriever.index(corpus_emb=corpus_emb, corpus_ids=corpus_ids)
            is_indexed = True

            del corpus_emb      # Release memory
        
        # Retrieve
        results = retriever.retrieve_with_emb(query_emb=query_emb, query_ids=query_ids, top_k=topk)

        # QA Sepcific: Annotate
        logger.info("Annotating...")
        start_time = time.time()
        answers_collection: Dict[str, List[str]] = answers_collections[lang]
        qrels: Dict[str, Dict[str, int]] = dict()     # qid -> {pid: is_revelent(1/0)}

        if pool is None:
            pool = Pool(12)
        
        if corpus_tokenized is None:
            # Pre-tokenize the whole corpus with Facebook Regex-based SimpleTokenizer
            # Processing can take a while
            corpus_tokenized = dict()
            corpus_tokenized_cache_path = Path(COLLECTION_PATH) / f"{os.path.splitext(CORPUS_FILENAME)[0]}.tok.cache.jsonl.gz"
            if corpus_tokenized_cache_path.exists():
                # Load from local tokenized cache
                with gzip.open(corpus_tokenized_cache_path, "rb") as f:
                    for line in tqdm(f, desc="Loading Pre-tokenized Corpus from Cache", total=len(corpus), mininterval=100):
                        item: dict = orjson.loads(line)
                        corpus_tokenized[item["id"]] = item["text_tokenized"]
            else:
                # Pre-tokenize
                rets = pool.map(
                    partial(tokenize_corpus, tokenizer=annotate_tokenizer),
                    iter_corpus(corpus),
                    chunksize=500,
                )
                with gzip.open(corpus_tokenized_cache_path, "wb") as f:
                    for item in tqdm(rets, desc="Pre-tokenizing Corpus", total=len(corpus), mininterval=100):
                        f.write(orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE))
                        corpus_tokenized[item["id"]] = item["text_tokenized"]            

        annotate_iter = iter_retrieval_scores(results, answers_collection=answers_collection, corpus_tokenized=corpus_tokenized)

        rets = pool.map(
            partial(annotate_func, tokenizer=annotate_tokenizer, regex=False),
            annotate_iter,
            chunksize=500,
        )
        for item in tqdm(rets, desc="Annotate", total=len(query_ids) * topk, mininterval=100):
            item: AnnotatePair
            if item.qid not in qrels:
                qrels[item.qid] = dict()
            qrels[item.qid][item.pid] = item.is_revelent
        
        logger.info(f"Annotation finished in {time.time()-start_time}s.")
            
        # Calculate metrics
        # Note:
        # For Retrieval of Open-domain QA tasks, there is a lack of 'closed qrels' because 
        # we are retrieve from 'open domain'. This means we actually do not know which passages
        # are ground truths until we retrieve and annotate them as revelent or irrevelent. 
        # Thus we usually measure the Accuarcy@k, which counts one retrieved passage as hit if Top-k 
        # retrieval results contains at least one correct answer. Then Accuarcy@k is obtained by 
        # averaging the number of hits@k over the number of queries.
        acc = top_k_accuracy(qrels, results, k_values)
        metric = dict()
        all_scores[split][lang] = dict()
        for k, v in acc.items():
            k: str
            if k.startswith("P@"):
                k = k.replace("P@", "precision_at_")
            k = k.replace("@", "_at_").lower()
            metric[k] = v
        
        all_scores[split][lang] = metric
        all_scores["evaluation_time"] = time.time() - init_time
        logger.info(f"--- Task: {task_name}{task_type}, Lang: {lang} ---")
        logger.info(metric)

        # Save metrics
        with open(score_path, 'wb') as f:
            f.write(orjson.dumps(all_scores, option=orjson.OPT_INDENT_2))
    
    logger.info(f"=== Evaluation Results ===")
    logger.info(all_scores)

    if pool is not None:
        pool.close()
        pool.join()
    
    logger.info("Done")


if __name__ == '__main__':

    def _parse_args() -> EvalArguments:
        # Wrap args parsing in this function to support type hint
        parser = HfArgumentParser(EvalArguments)
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            (args, ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            (args, ) = parser.parse_args_into_dataclasses()
        return args

    args = _parse_args()
    set_seed(args.seed)

    # Initialize RPC
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=f'tcp://{args.master_addr}:{args.master_port}',
        # Disable RDMA to avoid stuck
        # Ref: https://github.com/pytorch/tensorpipe/issues/457#issuecomment-1278720956
        _transports=[
            # "ibv",      # InfiniBand
            "shm",      # Shared Memory
            "uv",       # libuv - TCP
        ],
        _channels=[
            "cma",      # Cross-Memory Attach
            "mpt_uv",   # Multi-Protocol Transport over UV
            "basic", 
            # "cuda_gdr",     # InfiniBand
            "cuda_xth",     # CUDA Driver
            "cuda_ipc",     # CUDA Inter-Process Communication
            "cuda_basic"    # Basic CUDA API
        ],
    )
    rpc.init_rpc(
        name=f"worker{args.rank}",
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=rpc_backend_options,
    )
    logger.warning(f"[Rank{args.rank}] RPC Backends init successful.")

    # Local model on every ranks
    start_time = time.time()
    model = ExactSearchModel(args)
    logger.info(f"[Rank{args.rank}] Model loaded in {time.time()-start_time}s.")

    # Wait for model loading finished
    rpc.api._wait_all_workers(60)

    # Main process handles all data loading, processing, 
    # distribution, gathering encoded outputs and calculate
    # metrics
    if args.rank == 0:
        main(args, model)
    
    # Block until all RPCs are done
    rpc.shutdown()
