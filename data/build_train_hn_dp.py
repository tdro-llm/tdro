#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Examples for Mining Hard Negatives with Sentence Transformers Corpus, Query, Qrels.

This script bootstraps the following operations:
1. Read Corpus, Train Queries & Qrels.
2. Using Sentence Transformers to Encode Query & Passage Embeddings on a single node with DataParallel.
3. Faiss retrieve.
4. Produce Hard Negatives.

@Time    :   2024/03/04
@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import gzip
import time
import fire
import random
import orjson
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import chain
from functools import partial
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Union, Optional

import torch
from torch.utils.data import DataLoader

import faiss

import datasets
from datasets import Dataset, Value, load_dataset
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available, DataCollatorWithPadding, BatchEncoding

random.seed(42)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def build_corpus_idx_to_row(dataset: datasets.Dataset):
    """ Build a dict on memory of corpus id -> row id of hfdataset """
    idx_to_corpus = dict()
    for row_id, corpus_id in enumerate(dataset["_id"]):
        idx_to_corpus[corpus_id] = row_id
    return idx_to_corpus

def read_corpus(corpus_name_or_path: str):
    """ Load HFDataset from local or online """
    corpus_path = None
    # Local file or folders
    if os.path.exists(corpus_name_or_path):
        if os.path.isdir(corpus_name_or_path):
            files = os.listdir(corpus_name_or_path)
            corpus_path = [
                os.path.join(corpus_name_or_path, f)
                for f in files
                if f.endswith('json') or f.endswith('jsonl')
            ]
        else:
            corpus_path = [corpus_name_or_path]
        
        if corpus_path:
            # Load as Json files
            dataset_name = 'json'
            dataset_split = 'train'
            dataset_language = 'default'
        else:
            # Try to load from local HF dataset
            dataset_name = corpus_name_or_path
            dataset_split = 'train'
            dataset_language = None
            corpus_path = None
    # Online huggingface dataset
    else:
        info = corpus_name_or_path.split('/')
        dataset_split = info[-1] if len(info) == 3 else 'train'
        dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
        dataset_language = 'default'
        if ':' in dataset_name:
            dataset_name, dataset_language = dataset_name.split(':')
    
    dataset = load_dataset(
        dataset_name,
        dataset_language,
        data_files=corpus_path,
        split=dataset_split
    )

    # Parse tevatron format Jsonl text column names to sentence transformers format
    for _original_column_name, _new_column_name in [("query_id", "_id"), ("docid", "_id"), ("id", "_id"), ("query", "text"), ("question", "text")]:
        if _original_column_name in dataset.column_names:
            dataset = dataset.rename_column(_original_column_name, _new_column_name)
    
    # Format "_id" to str
    if "_id" in dataset.column_names and dataset.features["_id"].dtype != 'string':
        dataset = dataset.cast_column('_id', Value("string"))
    
    return dataset

@dataclass
class EncodeCollator(DataCollatorWithPadding):
    """
    DataCollator for processing & tokenize encode dataset.
    """
    prompt: str = ""
    separator: str = " "        # WhiteSpace
    
    def _get_passage(self, item: Dict[str, str]):
        if "title" in item:
            return item["title"] + self.separator + item["text"]
        else:
            return item["text"]

    def __call__(self, features: List[dict]):
        ids = list()
        texts = list()
        for item in features:
            ids.append(item["_id"])
            text = self._get_passage(item)
            if self.prompt:
                text = self.prompt + text
            texts.append(text)
        
        encoded: BatchEncoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation='only_first',
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        return ids, encoded


class BaseFaissIPRetriever:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class DenseEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.gpu_count = torch.cuda.device_count()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
        
        self.encoder = self.encoder.to(self.device)

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        
        self.encoder.eval()

    @torch.no_grad()
    def encode(
            self, 
            sentences: Union[List[str], List[Dict[str, any]], Dataset] , 
            batch_size=128, 
            max_length=512,
            **kwargs
        ) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        if isinstance(sentences, Dataset):
            dataset = sentences
        elif isinstance(sentences, list) and isinstance(sentences[0], dict):
            dataset: Dataset = Dataset.from_dict(sentences)
        elif isinstance(sentences, list) and isinstance(sentences[0], str):
            dataset: Dataset = Dataset.from_dict({'text': sentences})
        else:
            raise NotImplementedError

        collator = EncodeCollator(
            tokenizer=self.tokenizer,
            padding='longest',
            max_length=max_length,
        )
        encode_loader = DataLoader(
            dataset,
            batch_size=batch_size * self.gpu_count,
            collate_fn=collator,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

        encoded_embeds = []
        for _ids, batch_dict in tqdm(encode_loader, desc='encoding', mininterval=10, disable=len(sentences) < 128):
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                lm_out = self.encoder(**batch_dict)
                embeds: torch.Tensor = lm_out[0][:, 0]    # Perform pooling. In this case, cls pooling.
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)  # normalize embeddings
                encoded_embeds.append(embeds.cpu().numpy())
        
        del encode_loader

        return np.concatenate(encoded_embeds, axis=0)


# Build train HN with BEIR collections
def read_beir_collections(
    query_collection: str,          # Path to Queries. Jsonl Format: {"_id": str, "text": str}
    passage_collection: str,        # Path to Passage collections. Jsonl Format: {"_id": str, "title": str, "text": str}
    qrel_path: str,                 # Path to Train Query-revelent file. Tsv Format: "Query-id\tPassage-id"
):
    # Read qrels
    qrels_dataset = load_dataset(
        'csv',
        data_files=qrel_path,
        delimiter='\t',
        split="train",
    ).cast_column('query-id', Value("string")).cast_column('corpus-id', Value("string"))
    # Read corpus & queries
    queries: Dataset = read_corpus(query_collection).select_columns(["_id", "text"])
    corpus: Dataset = read_corpus(passage_collection).select_columns(["_id", "title", "text"])

    # Build str index -> hfdataset index
    idx_to_query: Dict[str, int] = build_corpus_idx_to_row(queries)        # Query-id -> Dataset index
    idx_to_passage: Dict[str, int] = build_corpus_idx_to_row(corpus)        # Corpus-id -> Dataset index

    # Format qrels: Query Dataset index -> List[Corpus Dataset index]
    qrels = defaultdict(list)
    for item in qrels_dataset:
        if item["corpus-id"] not in idx_to_passage:
            print(f"Escape of corpus-id {item['corpus-id']}, because it does not appear in the corpus.jsonl")
            continue
        _q_idx, _p_idx = idx_to_query[item["query-id"]], idx_to_passage[item["corpus-id"]]
        if _p_idx not in qrels[_q_idx]:
            qrels[_q_idx].append(_p_idx)

    print(f"# of Queries: {len(queries)}; # of Corpus: {len(corpus)} ")

    return corpus, queries, qrels


# This will read queries from provided training sets, and 
# read passages from positive passages and negative passages.
# Providing passage collection seperately is also ok.
def read_training_sets(
    train_collection: str,          # Path to Training Sets. Jsonl Format: {"_id": str, "text": str, "positive_passages": List[dict], "negative_passages": List[dict]}
    passage_collection: Optional[str] = None,        # Path to Passage collections. Jsonl Format: {"_id": str, "title": str, "text": str}
):
    # Read .gz format 
    train_ori_list = list()
    if train_collection.endswith(".gz"):
        with gzip.open(train_collection) as f:
            for idx, line in enumerate(tqdm(f, desc=f"Reading {train_collection}")):
                if idx > 5_000_000:     # Set max size mainly due to efficiency
                    break
                item = orjson.loads(line)
                # Convert Json list to Tevatron dict format if needed
                if isinstance(item, list):
                    # Pairs: ["text1", "text2"]
                    assert len(item) == 2, f"Num of fields for {train_collection} is: {item}. We only support json List with (query, answer), which has two fields."
                    formated_item = {
                            "_id": "train-" + str(idx),
                            "text": item[0],
                            "positive_passages": [
                                {
                                    "title": "",
                                    "text": item[1],
                                }
                            ]
                        }
                elif isinstance(item, dict):
                    if "query" in item and "pos" in item and "neg" not in item:
                        # Query-Pairs: {"query": "text", "pos": ["text1", "text2", ...]}
                        formated_item = {
                                "_id": "train-" + str(idx),
                                "text": item["query"],
                                "positive_passages": [
                                    {
                                        "title": "",
                                        "text": pos_text,
                                    } for pos_text in item["pos"] if pos_text.strip()
                                ]
                            }
                    elif "set" in item:
                        # Sets: {"set": ["text1", "text2", ...]}
                        assert isinstance(item["set"], list)
                        if len(item["set"]) < 2:
                            continue
                        formated_item = {
                                "_id": "train-" + str(idx),
                                "text": item["set"][0],
                                "positive_passages": [
                                    {
                                        "title": "",
                                        "text": pos_text,
                                    } for pos_text in item["set"][1:] if pos_text.strip()
                                ]
                            }
                    else:
                        raise NotImplementedError()
                if formated_item["text"] and formated_item["positive_passages"]:
                    train_ori_list.append(formated_item)
        train_ori: Dataset = Dataset.from_list(train_ori_list)
    else:
        # Read training sets
        train_ori: Dataset = read_corpus(train_collection)

    # If no "_id" appear in `queries` dataset, make "_id" by ourselves
    if "_id" not in train_ori.column_names:
        train_ori.add_column("_id", ["train-" + str(i) for i in range(len(train_ori))])
    
    # Cast some special datasets
    if "medmcqa" in train_collection:
        train_ori = train_ori.rename_column("question", "text")
        train_ori = train_ori.filter(lambda item: item["exp"], num_proc=4)
        train_ori = train_ori.map(lambda item: {"positive_passages": [{"title": "", "text": item["exp"]}]}, num_proc=4)
    
    # Format passages from string to dict
    def convert_liststr_to_listdict(item: List[str]):
        converted: List[Dict[str, str]] = []
        for psg in item:
            assert isinstance(psg, str)
            converted.append({"title": "", "text": psg})
        return converted
    
    if isinstance(train_ori[0]["positive_passages"][0], str):
        train_ori = train_ori.map(lambda x: {"positive_passages": convert_liststr_to_listdict(x["positive_passages"])}, num_proc=6)
    
    # Select all queries + positive_passages as training sets
    queries: Dataset = train_ori.select_columns(["_id", "text", "positive_passages"])

    # Collect passage collections
    if passage_collection is not None:  # Read from `passage_collection`
        corpus: Dataset = read_corpus(passage_collection)
        # If no "_id" appear in `corpus` dataset, make "_id" by ourselves
        if "_id" not in corpus.column_names:
            corpus.add_column("_id", [str(i) for i in range(len(queries))])
    else:   # Collect from `train_collection`
        corpus_list = list()
        corpus_content_set = set()
        for line in tqdm(train_ori, desc="Collecting corpus from training set"):
            for item in chain(line["positive_passages"], line.get("negative_passages", [])):
                title: str = item.get("title", "")
                text: str = item.get("text")
                
                if title + text not in corpus_content_set:
                    corpus_list.append(
                        {
                            "_id": item.get("_id", item.get("docid", str(len(corpus_list)))),
                            "title": title,
                            "text": text,
                        }
                    )
                    corpus_content_set.add(title + text)
        corpus: Dataset = Dataset.from_list(corpus_list)
                    
    # Build corpus `title + text` -> corpus row index
    corpus_content_to_idx: Dict[str, int] = dict()
    for idx, line in enumerate(tqdm(corpus, desc="Build corpus title + text -> row index")):
        title: str = line.get("title", "")
        text: str = line.get("text")
        corpus_content_to_idx[title + text] = idx
    
    # Collecting qrels
    qrels: Dict[int, List[int]] = defaultdict(list)
    for idx, line in enumerate(tqdm(queries, desc="Collecting qrels")):
        for pos_psg in line["positive_passages"]:
            title: str = pos_psg.get("title", "")
            text: str = pos_psg.get("text")
            qrels[idx].append(corpus_content_to_idx[title + text])
    
    return corpus, queries, qrels


def format_psg(item: dict, _idx: int):
    docid: int = item.get("_id", _idx)
    title: str = item.get("title", "")
    text: str = item["text"]
    return {
        "docid": str(docid),
        "title": title,
        "text": text
    }

def _sample_hn(
    input_item: Tuple[int, any],
    corpus: List[dict],
    queries: List[dict],
    qrels: Dict[str, List[str]],
    depth: List[int],    # Depth for Selecting Negatives.
    n_sample: int,              # Number of negatives to randomly select in the depth range.
) -> str:   # Return json serialized string
    time0 = time.time()
    q_idx, ranks = input_item
    time1 = time.time()
    neg_idxs: List[int] = [_idx for _idx in ranks[depth[0]: depth[1]] if _idx not in qrels[q_idx] and _idx >= 0]
    assert len(neg_idxs) > 0
    time2 = time.time()
    if len(neg_idxs) > n_sample:
        neg_idxs = random.sample(neg_idxs, n_sample)
    time3 = time.time()
    item: Dict[str, any] = {
        "query_id": queries[q_idx]["_id"],
        "query": queries[q_idx]["text"],
        "positive_passages": [format_psg(corpus[_idx], _idx) for _idx in qrels[q_idx]],    # List of {"_id": str, "title": str, "text": str}
        "negative_passages": [format_psg(corpus[_idx], _idx) for _idx in neg_idxs]         # List of {"_id": str, "title": str, "text": str}
    }
    time4 = time.time()
    line: bytes = orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE)
    time5 = time.time()
    return line, {"unpack input": time1-time0, "Get neg_idxs": time2-time1, "shuffle": time3-time2, "construct dict": time4-time3, "json": time5-time4}


def hn_mine(
    corpus: Dataset,
    queries: Dataset,
    qrels: Dict[str, List[str]],
    model_name_or_path: str,        # Path to Sentence Transformers model
    save_to: str,                   # Path to save training hard negatives.
    depth: List[int] = [2, 100],    # Depth for Selecting Negatives.
    topk: int = 100,                # Retrieve Top-k results with Faiss
    n_sample: int = 16,              # Number of negatives to randomly select in the depth range.
):
    # Load Model in DataParallel
    model = DenseEncoder(model_name_or_path)
    model.eval()

    # Encode Queries & Passages
    q_reps = model.encode(queries, batch_size=128)
    p_reps = model.encode(corpus, batch_size=128)

    # Create Faiss Index and Retrieve
    retriever = BaseFaissIPRetriever(embedding_dim=p_reps.shape[1])
    retriever.add(p_reps)

    # Loading to GPU, Multi-GPU for Faiss
    logger.info(f'Using multi GPUs for Faiss')
    co = faiss.GpuMultipleClonerOptions()
    co.shard, co.useFloat16 = True, False
    retriever.index = faiss.index_cpu_to_all_gpus(
        retriever.index,
        co=co,
    )

    # Search
    all_scores, all_indices = retriever.batch_search(q_reps, topk, batch_size=256, quiet=False)
    all_indices = all_indices.tolist()

    # Produce Hard Negatives
    assert len(all_indices) == q_reps.shape[0]
    save_to: Path = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)

    # debug
    time_records = defaultdict(list)
    _idx = 0

    with ThreadPoolExecutor() as p:
        with open(save_to, "wb") as f:
            rets = p.map(
                partial(_sample_hn, corpus=corpus.to_list(), queries=queries.to_list(), qrels=qrels, depth=depth, n_sample=n_sample),
                enumerate(tqdm(all_indices, desc="Sampling train-hn")),
                chunksize=500,
            )
            for line, all_time in tqdm(rets, desc="Writing train-hn", total=len(all_indices)):
                f.write(line)

                # Debug
                _idx += 1
                for k, v in all_time.items():
                    time_records[k].append(v)
                
                if _idx % 100000 == 0:
                    for k, v in time_records.items():
                        print(f"[{_idx}] {k} (s): {sum(v) / len(v)}")
                    print()


def mine_w_training_collection(
    train_collection: str,          # Path to Training Sets. Jsonl Format: {"_id": str, "text": str, "positive_passages": List[dict], "negative_passages": List[dict]}
    model_name_or_path: str,        # Path to Sentence Transformers model
    save_to: str,                   # Path to save training hard negatives.
    depth: List[int] = [2, 100],    # Depth for Selecting Negatives.
    topk: int = 100,                # Retrieve Top-k results with Faiss
    n_sample: int = 16,              # Number of negatives to randomly select in the depth range.
    passage_collection: Optional[str] = None,        # Path to Passage collections. Jsonl Format: {"_id": str, "title": str, "text": str}
):
    corpus, queries, qrels = read_training_sets(train_collection, passage_collection)
    hn_mine(corpus, queries, qrels, model_name_or_path, save_to, depth=depth, topk=topk, n_sample=n_sample)


def mine_w_qrels(
    query_collection: str,          # Path to Queries. Jsonl Format: {"_id": str, "text": str}
    passage_collection: str,        # Path to Passage collections. Jsonl Format: {"_id": str, "title": str, "text": str}
    qrel_path: str,                 # Path to Train Query-revelent file. Tsv Format: "Query-id\tPassage-id"
    model_name_or_path: str,        # Path to Sentence Transformers model
    save_to: str,                   # Path to save training hard negatives.
    depth: List[int] = [2, 100],    # Depth for Selecting Negatives.
    topk: int = 100,                # Retrieve Top-k results with Faiss
    n_sample: int = 16,              # Number of negatives to randomly select in the depth range.
):
    corpus, queries, qrels = read_beir_collections(query_collection, passage_collection, qrel_path)
    hn_mine(corpus, queries, qrels, model_name_or_path, save_to, depth=depth, topk=topk, n_sample=n_sample)


if __name__ == '__main__':
    fire.Fire(mine_w_training_collection)
