import os
import logging
import numpy as np
import queue
import threading
from threading import current_thread
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from tqdm import tqdm
from functools import partial
from dataclasses import dataclass, field
from typing import Mapping, List, Dict, Optional, Union, Tuple

# Store thread local variables (e.g. rank of current thread)
thread_local = threading.local()

import torch
import torch.utils.data
import torch.distributed
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase, BatchEncoding, PaddingStrategy
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import PeftModel, LoraConfig
from peft.utils import CONFIG_NAME as PEFT_CONFIG_NAME

from tdro.finetune.modeling_encoder import EncoderModel

from eval_arguments import EvalArguments

logger = logging.getLogger(__name__)

_MODEL_CLS = {
    "EncoderModel": EncoderModel,
}

@dataclass
class ExactSearchModel:
    """
    A Wrapper for EncoderModel.
    
    Exact Search (ES) in BeIR requires an encode_queries & encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into BeIR DRES format.
    """
    args: EvalArguments

    # Only used by the RPC main process
    query_prompt: Optional[str] = None
    corpus_prompt: Optional[str] = None
    encoding_kwargs: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.args.world_size > 0 and self.args.local_rank >= 0, \
            "Please use rpc encoding with torchrun"

        # Process input/output data, send encoding message only in main rank
        if self.args.local_rank == 0:
            self.executor = ThreadPoolExecutor(max_workers=self.args.world_size, initializer=self._init_thread_local_variables, initargs=(list(range(self.args.world_size)),))

        self.tokenizer = _load_tokenizer(self.args.model_name_or_path)
        self.model = self._load_model(target_device=self.args.local_rank, args=self.args)
        self.model.eval()
        if self.model.lm_q.device == torch.device("cpu"):
            self.model.to(self.args.local_rank)
        
        global MODEL_REGISTRY
        MODEL_REGISTRY = {"worker": self}

    def encode_queries(self, queries: List[str], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> torch.Tensor:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            queries (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """        
        if self.query_prompt:
            queries = [self.query_prompt + i for i in queries]
        
        return self._encode(queries, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, tokenizer=self.tokenizer, max_length=self.args.q_max_len, encode_is_query=True, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> torch.Tensor:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            corpus (`List[str]` or `List[Dict[str, str]]`): List of sentences to encode
                or list of dictionaries with keys "title" and "text"
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.args.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        elif type(corpus) is list and type(corpus[0]) is dict:
            sentences = [
                (doc["title"] + self.args.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        elif type(corpus) is list and type(corpus[0]) is str:
            sentences = corpus
        else:
            raise NotImplementedError()

        return self.encode(sentences, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, **kwargs)
    
    def encode(self, sentences: List[str], batch_size: int, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """        
        if self.corpus_prompt:
            sentences = [self.corpus_prompt + i for i in sentences]
        
        return self._encode(sentences, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, tokenizer=self.tokenizer, max_length=self.args.p_max_len, encode_is_query=False, **kwargs)
    
    def _encode(
        self, 
        sentences: List[str], 
        batch_size: int, 
        show_progress_bar: bool, 
        convert_to_tensor: bool, 
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        encode_is_query: bool,
        **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        data_iter = torch.utils.data.DataLoader(
            sentences, 
            batch_size=batch_size, 
            # Python Multi-Processing duplicates in-memory dataset.
            # See: https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
            #
            # TODO: I cannot see speed improvement when using multiprocess dataloader with in-memory 
            # string list
            num_workers=0,
            # We do not know which GPU to copy (GPU may even be cross-node) when using RPC Encoding.
            pin_memory=False,
            collate_fn=EncodeCollator(tokenizer, padding=self.args.padding, max_length=max_length, pad_to_multiple_of=self.args.pad_to_multiple_of),
        )
        data_iter_pbar = tqdm(
            data_iter, 
            total=len(data_iter), 
            mininterval=10, 
            disable=(not show_progress_bar), 
            desc=f"Encoding in-queue {'query' if encode_is_query else 'corpus'} [n_gpu={self.args.world_size}, bs={batch_size}]"
        )

        if self.args.debug:
            # Debug encoding function on rank0
            rets = map(
                partial(self._rpc_encode, encode_is_query=encode_is_query, encoding_kwargs=self.encoding_kwargs), 
                data_iter_pbar
            )
        else:
            rets = self.executor.map(
                partial(self._submit, encode_is_query=encode_is_query, encoding_kwargs=self.encoding_kwargs), 
                data_iter_pbar
            )
        
        encoded_embeds: dict[str, list[Tensor | str | dict]] = dict()
        is_multi_vector_retrieval = False
        for emb in tqdm(
            rets, 
            total=len(data_iter), 
            mininterval=10, 
            disable=(not show_progress_bar), 
            desc=f"Encoding out-queue {'query' if encode_is_query else 'corpus'} [n_gpu={self.args.world_size}, bs={batch_size}]"
        ):
            if isinstance(emb, (torch.Tensor, np.ndarray)):   # Dense
                if 'dense_reps' not in encoded_embeds:
                    encoded_embeds['dense_reps'] = list()
                encoded_embeds['dense_reps'].append(emb)
            
            elif isinstance(emb, dict):   # Hybrid
                is_multi_vector_retrieval = True
                for k, v in emb.items():
                    if k not in encoded_embeds:
                        encoded_embeds[k] = list()
                    if isinstance(v, (torch.Tensor, np.ndarray)):   # Dense
                        encoded_embeds[k].append(v)
                    elif isinstance(v, list):   # Sparse
                        encoded_embeds[k].extend(v)

        # Concat Tensor
        logger.info("Collecting encoded embeddings...")
        if is_multi_vector_retrieval:
            for k in encoded_embeds.keys():
                if isinstance(encoded_embeds[k][0], (torch.Tensor, np.ndarray)):
                    encoded_embeds[k] = torch.cat(encoded_embeds[k], dim=0)
                    if not convert_to_tensor:
                        encoded_embeds[k] = encoded_embeds[k].numpy()
        else:
            encoded_embeds = torch.cat(encoded_embeds['dense_reps'], dim=0)
            if not convert_to_tensor:
                encoded_embeds = encoded_embeds.numpy()
        
        # Clear up cache
        self.empty_cache()

        return encoded_embeds

    
    @staticmethod
    def _compile_model(model: EncoderModel):
        if (not model.model_args.untie_encoder) and (id(model.lm_q) == id(model.lm_p)):
            model.lm_q.forward = torch.compile(model.lm_q.forward)   # options={"triton.cudagraphs": True} mode="max-autotune"
            model.lm_p = model.lm_q
        else:
            model.lm_q.forward = torch.compile(model.lm_q.forward)
            model.lm_p.forward = torch.compile(model.lm_p.forward)
        torch.cuda.empty_cache()
        return model
    
    @classmethod
    def _load_model(cls, target_device: int, args: EvalArguments):
        """ 
        Load an EncoderModel (which can also be ExactSearchModel.model)
        """
        # Init Model & Load to `target_device`
        model: EncoderModel = _MODEL_CLS[args.model_type].load(
            model_name_or_path=args.model_name_or_path,
            model_args=args,
            # HF Argument
            attn_implementation=args.attn_implementation,
            torch_dtype=args.dtype,
            device_map=target_device,
        )

        # Compile model if PyTorch 2.x
        if args.torch_compile and torch.__version__.split(".")[0] >= "2":
            logging.info(f"[{target_device}] Compiling model..")
            model = cls._compile_model(model)
            logging.info(f"[{target_device}] Torch Compile Complete!")
        
        return model

    @staticmethod
    def _init_thread_local_variables(ranks: list[int]):
        """ 
        Receives a shared list of ranks, and pop the first element as 
        the rank of current thread 
        """
        thread_local.rank = ranks.pop(0)
        # store the unique worker name in a thread local variable
        logging.warning(f'[Rank{thread_local.rank}] Initializing worker thread {current_thread().name}')

    @classmethod
    def _submit(cls, batch: dict | BatchEncoding, encode_is_query: bool, encoding_kwargs: dict):
        """ 
        Submit a encoding task to one thread in the `main rank`.
        This function is executed on each threads of main rank.
        """
        assert isinstance(batch, (dict, BatchEncoding))
        assert isinstance(encode_is_query, bool)
        assert isinstance(encoding_kwargs, dict)

        embeddings = rpc.rpc_sync(
            thread_local.rank, 
            cls._rpc_encode, 
            kwargs={
                "batch": batch,
                "encode_is_query": encode_is_query,
                "encoding_kwargs": encoding_kwargs,
            }
        )
        return embeddings
    
    @staticmethod
    def _rpc_encode(batch: dict | BatchEncoding, encode_is_query: bool, encoding_kwargs: dict):
        """
        Encode warpper function used by each `worker`
        This function is executed on workers.
        """
        # Global registry (global environment variable) seems to be the only way 
        # we can get the pointer of model during remote rpc execution.
        global MODEL_REGISTRY
        model = MODEL_REGISTRY['worker'].model

        batch = move_to_cuda(batch, device=model.lm_q.device)
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            if encode_is_query:
                embeddings: Union[Tensor, Dict[str, Tensor]] = model.encode_query(qry=batch, **encoding_kwargs)
            else:
                embeddings: Union[Tensor, Dict[str, Tensor]] = model.encode_passage(psg=batch, **encoding_kwargs)
        
        if isinstance(embeddings, Tensor):
            embeddings = embeddings.cpu()
        elif isinstance(embeddings, dict):
            for k in embeddings.keys():
                if embeddings[k] is not None:
                    embeddings[k] = embeddings[k].cpu()
        else:
            raise NotImplementedError()

        return embeddings
    
    def empty_cache(self):
        for rank in range(self.args.world_size):
            rpc.rpc_sync(rank, torch.cuda.empty_cache)


@dataclass
class EncodeCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: str = 'only_first'
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, texts: List[str]) -> BatchEncoding:
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            # return_token_type_ids=False,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

def move_to_cuda(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if isinstance(maybe_tensor, Tensor):
            return maybe_tensor.to(device)
            # # Set CUDA_VISIBLE_DEVICES or torch.cuda.set_device() 
            # # before using .cuda()
            # return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

# Tokenizer
def _load_tokenizer(model_name_or_path: str):
    if model_name_or_path is None:
        return None
    
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name_or_path)

    # Compatiable with GPT Tokenizers
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})   # Always add a pad token
    tokenizer.padding_side = "right"
    if hasattr(tokenizer, "add_eos_token") and tokenizer.add_eos_token == False:
        # Add </eos> for compatiable with last token pooling
        # Note: This is working for Pythia, LLaMA2 & Mistral, but not LLaMA3, Qwen, Phi. We need to modify the 
        # `post_processor` of tokenizer.json for them to add </eos>
        tokenizer.add_eos_token = True
    
    return tokenizer


def get_task_def_by_task_name_and_type_e5(task_name: str, task_type: str) -> str:
    if task_type in ['STS']:
        return "Retrieve semantically similar text."

    if task_type in ['Summarization']:
        return "Given a news summary, retrieve other semantically similar summaries"

    if task_type in ['BitextMining']:
        return "Retrieve parallel sentences."

    if task_type in ['Classification']:
        task_name_to_instruct: Dict[str, str] = {
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classify the given Amazon review into its appropriate rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
            # C-MTEB eval instructions
            'TNews': 'Classify the fine-grained category of the given news title',
            'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
            'MultilingualSentiment': 'Classify sentiment of the customer review into positive, neutral, or negative',
            'JDReview': 'Classify the customer review for iPhone on e-commerce platform into positive or negative',
            'OnlineShopping': 'Classify the customer review for online shopping into positive or negative',
            'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Clustering']:
        task_name_to_instruct: Dict[str, str] = {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
            # C-MTEB eval instructions
            'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
            'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
            'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
            'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Reranking', 'PairClassification']:
        task_name_to_instruct: Dict[str, str] = {
            'AskUbuntuDupQuestions': 'Retrieve duplicate questions from AskUbuntu forum',
            'MindSmallReranking': 'Retrieve relevant news articles based on user browsing history',
            'SciDocsRR': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
            'StackOverflowDupQuestions': 'Retrieve duplicate questions from StackOverflow forum',
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum',
            'TwitterSemEval2015': 'Retrieve tweets that are semantically similar to the given tweet',
            'TwitterURLCorpus': 'Retrieve tweets that are semantically similar to the given tweet',
            # C-MTEB eval instructions
            'T2Reranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoReranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CMedQAv1': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'CMedQAv2': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'Ocnli': 'Retrieve semantically similar text.',
            'Cmnli': 'Retrieve semantically similar text.',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Retrieval']:
        if task_name.lower().startswith('cqadupstack'):
            return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'

        task_name_to_instruct: Dict[str, str] = {
            'ArguAna': 'Given a claim, find documents that refute the claim',
            'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim',
            'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia',
            'FEVER': 'Given a claim, retrieve documents that support or refute the claim',
            'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question',
            'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question',
            'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query',
            'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question',
            'NQ': 'Given a question, retrieve Wikipedia passages that answer the question',
            'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question',
            'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
            'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim',
            'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
            'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
            # C-MTEB eval instructions
            'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
            'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
            'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
            'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
            'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos',
            # MIRACL
            'MIRACLRetrieval': 'Given a question, retrieve Wikipedia passages that answer the question',
            # MLDR
            'MultiLongDocRetrieval': 'Given a question, retrieve documents that answer the question',
            # MKQA Test
            "MKQA": "Given a question, retrieve Wikipedia passages that answer the question",
        }

        # add lower case keys to match some beir names
        task_name_to_instruct.update({k.lower(): v for k, v in task_name_to_instruct.items()})
        # other cases where lower case match still doesn't work
        task_name_to_instruct['trec-covid'] = task_name_to_instruct['TRECCOVID']
        task_name_to_instruct['climate-fever'] = task_name_to_instruct['ClimateFEVER']
        task_name_to_instruct['dbpedia-entity'] = task_name_to_instruct['DBPedia']
        task_name_to_instruct['webis-touche2020'] = task_name_to_instruct['Touche2020']
        task_name_to_instruct['fiqa'] = task_name_to_instruct['FiQA2018']
        task_name_to_instruct['quora'] = task_name_to_instruct['QuoraRetrieval']

        return task_name_to_instruct[task_name]

    raise ValueError(f"No instruction config for task {task_name} with type {task_type}")

def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)

def get_mteb_prompt(task_name: str, task_type: str, prompt_type: str):
    if prompt_type == "e5_small_model":     # Special case for e5_small_model
        if task_type in ["Reranking", "Retrieval"]:
            query_prompt = "query: "
            corpus_prompt = "passage: "
        else:
            query_prompt = "query: "
            corpus_prompt = "query: "
    elif prompt_type == 'e5':
        instruct = get_task_def_by_task_name_and_type_e5(task_name, task_type)
        query_prompt = get_detailed_instruct(instruct)
        if task_type in ["Reranking", "Retrieval"]:
            corpus_prompt = ""
        else:
            corpus_prompt = query_prompt    # Actually Unused
    elif prompt_type == 'bge':
        if task_type in ['Retrieval']:
            query_prompt = "Represent this sentence for searching relevant passages: "
        else:
            query_prompt = ""
        
        if task_type in ["Reranking", "Retrieval"]:
            corpus_prompt = ""
        else:
            corpus_prompt = query_prompt    # Actually Unused
    else:
        raise NotImplementedError()
    
    return query_prompt, corpus_prompt
