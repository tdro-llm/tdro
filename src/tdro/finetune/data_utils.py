#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training datasets.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''
import random
from dataclasses import dataclass
from itertools import chain
from typing import List, Tuple, Dict, Optional, Union

import datasets
from transformers import PreTrainedTokenizerBase, BatchEncoding, DataCollatorWithPadding

import torch
from torch.utils.data import Dataset

from .arguments import DataArguments
from .trainer import ContrastiveTrainer
from ..utils.data_utils import read_corpus, build_corpus_idx_to_row

import logging
logger = logging.getLogger(__name__)

INSTS = {
    "e5": {
        # Retrieval
        "agnews": ["Given a news title, retrieve the news descriptions that match the title"],
        "AllNLI": ["Given a premise, retrieve a hypothesis that is entailed by the premise", "Retrieve semantically similar text."],
        "altlex": ["Given a sentence, retrieve a paraphrase Wikipedia sentence", "Given a passage, retrieve a Wikipedia passage that forms paraphrase pairs"],
        "amazon-qa": ["Given a question, retrieve the corresponding answers from Amazon", "Given a question, retrieve an Amazon answer that solves the question"],
        "amazon_review_2018": ["Given a title, retrieve the corresponding reviews from Amazon", "Given a title, retrieve a Amazon review"],
        "amazon_review_2018_1m": ["Given a title, retrieve the corresponding reviews from Amazon", "Given a title, retrieve a Amazon review"],
        "ccnews_title_text": ["Given a news title, retrieve articles that match the title"],
        "cnn_dailymail": ["Given highlight sentences, retrieve an relevant article that match the sentences"],
        "cnn_dailymail_splitted": ["Given a news article, retrieve its highlight sentences", "Given a passage, retrieve the corresponding highlight sentences"],
        "coco_captions": ["Given a caption, retrieve a similar caption from the same image", "Given a caption, retrieve a caption that describes the same image"],
        "codesearchnet": ["Given a comment of the function code, retrieve the corresponding code blocks"],
        "dureader": ["Given a Chinese search query, retrieve web passages that answer the question"],
        "eli5_question_answer": ["Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum"],
        "fever": ["Given a claim, retrieve documents that support or refute the claim"],
        "flickr30k_captions": ["Given a caption, retrieve a similar caption from the same image", "Given a caption, retrieve a caption that describes the same image"],
        "gooaq_pairs": ["Given a web search query, retrieve the corresponding answers from Google"],
        "hotpotqa": ["Given a multi-hop question, retrieve documents that can help answer the question"],
        "medmcqa": ["Given a medical query, retrieve relevant passages that answer the query", "Given a medical question, retrieve passages that answer the question"],
        "miracl": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "MLDR": ["Given a question, retrieve documents that answer the question", "Retrieve documents that answer the question"],
        "mr_tydi_combined": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "msmarco": ["Given a web search query, retrieve relevant passages that answer the query"],
        "npr": ["Given a news title, retrieve articles that match the title"],
        "nq": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "PAQ_pairs": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "PAQ_pairs_100k": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "quora_duplicates_triplets": ["Given a question, retrieve questions that are semantically equivalent to the given question", "Find questions that have the same meaning as the input question"],
        "S2ORC_title_abstract": ["Given a title, retrieve the abstract from scientific papers", "Given a title, retrieve abstracts from scientific papers that match the title"],
        "S2ORC_title_abstract_100k": ["Given a title, retrieve the abstract from scientific papers", "Given a title, retrieve abstracts from scientific papers that match the title"],
        "searchQA_top5_snippets": ["Given a question, retrieve text snippets that answer the question", "Retrieve text snippets that answer the question"],
        "sentence-compression": ["Given a sentence, retrieve a short sentence that is semantically equivalent to the given sentence"],
        "SimpleWiki": ["Given a Wikipedia sentence, retrieve sentences that are semantically equivalent to the given sentence", "Retrieve semantically similar text."],
        "specter_train_triples": ["Given a title, retrieve semantic related titles", "Retrieve semantic related titles from scientific publications"],
        "squad_pairs": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "stackexchange_duplicate_questions_body_body": ["Retrieve duplicate passages from StackOverflow forum"],
        "stackexchange_duplicate_questions_title-body_title-body": ["Retrieve duplicate questions and passages from StackOverflow forum"],
        "stackexchange_duplicate_questions_title_title": ["Retrieve duplicate questions from StackOverflow forum"],
        "t2ranking": ["Given a Chinese search query, retrieve web passages that answer the question"],
        "trivia": ["Given a question, retrieve Wikipedia passages that answer the question", "Retrieve Wikipedia passages that answer the question"],
        "WikiAnswers": ["Retrieve duplicate questions from Wikipedia"],
        "WikiAnswers_100k": ["Retrieve duplicate questions from Wikipedia"],
        "wikihow": ["Given a summary, retrieve Wikipedia passages that match the summary"],
        "xsum": ["Given a news summary, retrieve articles that match the summary"],
        "yahoo_answers_question_answer": ["Given a question, retrieve Yahoo answers that solve the question"],
        "yahoo_answers_title_answer": ["Given a title, retrieve Yahoo answers that match the title"],
        "yahoo_answers_title_question": ["Given a title, retrieve the corresponding Yahoo questions"],

        # MKQA
        "MKQA": ["Given a question, retrieve Wikipedia passages that answer the question"],
    },
}


class TrainDataset(Dataset):
    """ Wrapper for Sampling Positive / Negative Passages """
    def __init__(
            self,
            data_args: DataArguments,
            dataset: str,                                  # String Path to Training Triples with Negatives
            query_collection: Optional[str] = None,        # String Path to query corpus
            passage_collection: Optional[str] = None,      # String Path to passage corpus
            trainer: ContrastiveTrainer = None,
            train_n_passages: int = 8,
            positive_passage_no_shuffle: bool = False,
            negative_passage_no_shuffle: bool = False,
    ):
        self.train_data = read_corpus(dataset)
        self.trainer = trainer
        self.data_args = data_args
        self.train_n_passages = train_n_passages
        self.positive_passage_no_shuffle = positive_passage_no_shuffle
        self.negative_passage_no_shuffle = negative_passage_no_shuffle

        self.read_text_from_collections = (query_collection is not None) and (passage_collection is not None)
        if query_collection is not None:
            # Load query corpus
            self.query_dataset: datasets.Dataset = read_corpus(query_collection)
            self.idx_to_query: Dict[str, int] = build_corpus_idx_to_row(self.query_dataset)
        
        if passage_collection is not None:
            # Load passage corpus
            self.passage_dataset: datasets.Dataset = read_corpus(passage_collection)
            self.idx_to_passage: Dict[str, int] = build_corpus_idx_to_row(self.passage_dataset)
    
    def get_query(self, _id: str) -> dict:
        return self.query_dataset[self.idx_to_query[_id]]
    
    def get_passage(self, _id: str) -> dict:
        return self.passage_dataset[self.idx_to_passage[_id]]
    
    def __len__(self):
        return len(self.train_data) 

    def __getitem__(self, index: int) -> Dict[str, any]:
        group = self.train_data[index]
        _hashed_seed = hash(index + self.trainer.args.seed)

        epoch = int(self.trainer.state.epoch)

        # Read Query
        if self.read_text_from_collections:
            qry: str = self.get_query(group['_id'])['text']
        else:
            qry: str = group['text']

        # Sample One Positive
        group_positives = group['positive_passages']
        if self.positive_passage_no_shuffle:
            pos_psg: Dict[str, any] = group_positives[0]
        else:
            pos_psg: Dict[str, any] = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        if self.read_text_from_collections:
            pos_psg.update(self.get_passage(pos_psg['docid']))

        # Sample Negatives
        group_negatives = group['negative_passages']
        negative_size = self.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.train_n_passages == 1:
            negs = []
        elif self.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]
        
        if self.read_text_from_collections:
            negs_w_texts = list()
            for item in negs:
                item.update(self.get_passage(item['docid']))
                negs_w_texts.append(item)
            negs = negs_w_texts

        return {
            "query": qry,
            "positive_passages": [pos_psg],
            "negative_passages": negs,
        }


@dataclass
class TrainCollator(DataCollatorWithPadding):
    """
    DataCollator for processing & tokenize train dataset.
    """
    q_max_len: int = 512
    p_max_len: int = 512
    # separator: str = getattr(self.tokenizer, "sep_token", ' ')  # [SEP]
    separator: str = " "        # WhiteSpace
    query_tokenizer: PreTrainedTokenizerBase = None

    def __post_init__(self):
        if self.query_tokenizer is None:
            self.query_tokenizer = self.tokenizer
    
    def _get_query(self, item: Dict[str, str]) -> str:
        query = item["query"]
        if "query_prompt" in item:
            query = item["query_prompt"] + query
        
        return query
    
    def _get_passages(self, item: Dict[str, Dict[str, str]]) -> List[str]:
        assert isinstance(item["positive_passages"], list) and isinstance(item["negative_passages"], list)
        assert len(item["positive_passages"]) == 1, f"Contrastive learning needs 1 positive passage, but found {len(item['positive_passages'])}"

        all_psgs: List[str] = []
        for psg in chain(item["positive_passages"], item["negative_passages"]):
            if "title" in psg:
                text = psg["title"] + self.separator + psg["text"]
            else:
                text = psg["text"]
            
            if "passage_prompt" in item:
                text = item["passage_prompt"] + text
            
            all_psgs.append(text)
        
        return all_psgs

    def __call__(self, features: List[dict]):
        # Tokenize `Query`
        q_texts: List[str] = list(map(self._get_query, features))
        q_tokenized: BatchEncoding = self.query_tokenizer(
            q_texts,
            max_length=self.q_max_len,
            truncation='only_first',
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=self.return_tensors,
        )

        # Process `Passage` & `Negatives`
        p_texts: List[str] = sum(map(self._get_passages, features), [])
        
        # Tokenize Passage
        p_tokenized: BatchEncoding = self.tokenizer(
            p_texts,
            max_length=self.p_max_len,
            truncation='only_first',
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=self.return_tensors,
        )

        processed = {
            "query": q_tokenized,
            "passage": p_tokenized,
        }
        
        # Sample CE Scores for Distillation
        if 'ce_score' in features[0]["positive_passages"][0]:
            ce_scores: List[float] = []
            for item in features:
                ce_scores.append(float(item["positive_passages"][0]['ce_score']))
                for _neg in item["negative_passages"]:
                    ce_scores.append(float(_neg['ce_score']))
            processed["ce_scores"] = torch.tensor(ce_scores, dtype=torch.float32)

        if "domain_ids" in features[0]:
            processed["domain_ids"] = torch.tensor([features[i]["domain_ids"] for i in range(len(features))], dtype=torch.int64)
        
        if "domain_name" in features[0]:
            processed["domain_name"] = [features[i]["domain_name"] for i in range(len(features))]    

            # Support masking in-batch / cross-batch negatives when encounter special tasks
            task_prefixs_for_only_hn = ["clustering", "classification"]
            only_hn: List[bool] = []
            for item in features:
                # Only use hard negatives, do not use in-batch / cross-batch negatives
                if any(_prefix in item["domain_name"] for _prefix in task_prefixs_for_only_hn):
                    only_hn.append(True)
                else:
                    only_hn.append(False)
            processed["only_hn"] = torch.tensor(only_hn, dtype=torch.bool)
        
        return processed


@dataclass
class IterableTrainCollator(TrainCollator):
    """
    IterableTrainCollator for sample batch examples, processing & tokenize train dataset.
    """
    train_n_passages: int = 2
    seed: int = 42
    positive_passage_no_shuffle: bool = False
    negative_passage_no_shuffle: bool = False
    add_prompt_prob: float = -1.
    prompt_type: str = 'e5'

    def __post_init__(self):
        super(IterableTrainCollator, self).__post_init__()
        self.rng = random.Random(self.seed)
    
    def __call__(self, group: List[dict]):
        return super(IterableTrainCollator, self).__call__(list(map(self.get_item, group)))
    
    def get_item(self, group: dict):
        # Sample One Positive
        group_positives = group['positive_passages']
        if self.positive_passage_no_shuffle:
            pos_psg: Dict[str, any] = group_positives[0]
        else:
            pos_psg: Dict[str, any] = self.rng.choice(group_positives)
        
        # Sample Negatives
        group_negatives = group['negative_passages']
        negative_size = self.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = self.rng.choices(group_negatives, k=negative_size)
        else:
            if self.train_n_passages == 1:
                negs = []
            elif self.negative_passage_no_shuffle:
                negs = group_negatives[:negative_size]
            else:
                negs = self.rng.sample(group_negatives, k=negative_size)
        
        rets = {
            "query": group["query"],
            "positive_passages": [pos_psg],
            "negative_passages": negs,
            "domain_name": group["domain_name"],
        }

        if "domain_ids" in group:
            rets["domain_ids"] = group["domain_ids"]
        
        if self.add_prompt_prob > 0 and self.add_prompt_prob <= 1:
            if self.add_prompt_prob >= 1.0 or self.rng.random() <= self.add_prompt_prob:    # Speed up when add_prompt_prob >= 1.0
                rets["query_prompt"] = self.get_prompt(task_name=group["domain_name"])
                if any(i in group["domain_name"] for i in ["NLI", "altlex", "captions", "duplicate", "SimpleWiki", "specter_train_triples", "WikiAnswers"]):
                    # Also add prompt to every passages for Symmetrical tasks
                    rets["passage_prompt"] = self.get_prompt(task_name=group["domain_name"])

        return rets

    def get_prompt(self, task_name: str) -> str:
        """ Add prompt for query of QA/Retrieval tasks. 
            Also add prompt on both side for Symmetrical tasks (e.g. NLI) 
        """
        if self.prompt_type == 'e5':
            instruct_list: List[str] = INSTS[self.prompt_type][task_name]
            instruct: str = instruct_list[0] if len(instruct_list) == 1 else self.rng.choice(instruct_list)
            prompt = 'Instruct: {}\nQuery: '.format(instruct) if instruct != '' else ''
            return prompt
        elif self.prompt_type == 'bge':
            if any(i in task_name for i in ["NLI", "altlex", "captions", "duplicate", "SimpleWiki", "specter_train_triples", "WikiAnswers"]):
                # No need to add prompt which is not retrieval tasks
                return ""
            else:
                # Only add query prompt to **Retrieval Tasks**
                return "Represent this sentence for searching relevant passages: "
        else:
            raise NotImplementedError()


def get_encode_prompt(task_name: str, prompt_type: str):
    if prompt_type == 'e5':
        instruct_list: List[str] = INSTS[prompt_type][task_name]
        assert len(instruct_list) == 1, f"Encoding {task_name} needs only one instruct."
        instruct: str = instruct_list[0]
        prompt = 'Instruct: {}\nQuery: '.format(instruct) if instruct != '' else ''
        return prompt
    elif prompt_type == 'bge':
        if any(i in task_name for i in ["NLI", "altlex", "captions", "duplicate", "SimpleWiki", "specter_train_triples", "WikiAnswers"]):
            # No need to add prompt which is not retrieval tasks
            return ""
        else:
            # Only add query prompt to **Retrieval Tasks**
            return "Represent this sentence for searching relevant passages: "
    else:
        raise NotImplementedError()

@dataclass
class EncodeCollator(DataCollatorWithPadding):
    """
    DataCollator for processing & tokenize encode dataset.
    """
    prompt: str = ""
    # separator: str = getattr(self.tokenizer, "sep_token", ' ')  # [SEP]
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
