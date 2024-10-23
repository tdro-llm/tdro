#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training arguments.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
from transformers import TrainingArguments

@dataclass
class DomainConfig:
    epoch: dict[str, float] | None = field(
        default=None, metadata={"help": "Domain name -> reference epoch."}
    )
    domain_weights: dict[str, float] | None = field(
        default=None, metadata={"help": "Domain name -> domain weights."}
    )
    n_groups: int | None = field(
        default=None, metadata={"help": "Number of domains / groups."}
    )
    domain_ids: dict[str, int] | None = field(
        default=None, metadata={"help": "Domain name -> domain id."}
    )
    size: dict[str, int] | None = field(
        default=None, metadata={"help": "Domain name -> Number of item/lines in this domain."}
    )
    category_list: dict[str, list[str]] | None = field(
        default=None, metadata={"help": "Category name -> List of domain names. This is optionally used to group the domain from the same category"}
    )

    # Set some alias
    @property
    def domain_to_idx(self):
        return self.domain_ids
    
    @property
    def domain_size(self):
        return self.size
    
    def __post_init__(self):
        if (self.domain_weights is not None) and (self.n_groups is None):
            self.n_groups = len(self.domain_weights)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    query_collection: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    passage_collection: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    corpus_path: Optional[str] = field(
        default=None, metadata={"help": "Path to train triples / encode corpus data"}
    )
    dev_path: Optional[str] = field(
        default=None, metadata={"help": "Path to development triples, the same format as training negative triples."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    q_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: Union[bool] = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    pad_to_multiple_of: Optional[int] = field(default=None)

    train_n_passages: int = field(default=8)
    eval_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    qrel_path: Optional[str] = field(
        default=None, metadata={"help": "Path to qrels for filtering out queries to encode."}
    )
    encoded_save_prefix: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)

    # Interleavable datasets
    domain_config_path: Optional[str] = field(
        default=None, metadata={"help": "Path to json format domain config."}
    )
    preprocessed_dir: Optional[str] = field(
        default=None, metadata={"help": "Root folder path of all processed domains."}
    )
    add_domain_id: bool = field(
        default=True, metadata={"help": "Add domain index."}
    )
    add_domain_name: bool = field(
        default=True, metadata={"help": "Add domain name. This helps to add instruct to query."}
    )
    add_prompt: bool = field(
        default=False, metadata={"help": "Add prompt for training. Please explicitly setting this to true if you want to add prompt to the query side."}
    )
    add_prompt_prob: float = field(
        default=1.0, metadata={"help": "Probality to add prompt to query, range (0, 1]."}
    )
    prompt_type: str = field(
        default="e5", metadata={"help": "Choosing among 'e5', 'instructor', 'bge'."}
    )
    encoding_task_name: Optional[str] = field(
        default=None, metadata={"help": "Task name used during the inference."}
    )
    stopping_strategy: str = field(
        default="all_exhausted", metadata={"help": "Set to 'first_exhausted' for less sampling "
                                "or 'all_exhausted' for oversampling."
                                "See `datasets.interleave_datasets`"}
    )

    homogenous_batch: bool = field(
        default=False, metadata={"help": "Yeilds a homogenous batch from one dataset at each iteration."}
    )

    domain_config: DomainConfig | None = field(
        default=None, metadata={"help": "Domain config init from `domain_config_path`."}
    )

    def __post_init__(self):
        if (self.domain_config_path is not None) and (self.domain_config is None):
            # Load domain weights from local file
            with open(self.domain_config_path, 'r') as f:
                domain_config: dict = json.load(f)
                self.domain_config = DomainConfig(**domain_config)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
                    "This will override both `model_name_or_path_qry` and `model_name_or_path_psg`."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model archeticture used in training."
                    "Choose among ['EncoderModel']."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    add_pooler: bool = field(
        default=False,
        metadata={"help": "Add a MLP layer on top of pooled embedding."}
    )
    projection_in_dim: int = field(
        default=None,
        metadata={"help": "MLP Fan-in."}
    )
    projection_out_dim: int = field(
        default=None,
        metadata={"help": "MLP Fan-out."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    cumulative_seq: bool = field(
        default=False, 
        metadata={
            "help": "Whether to use automatic cumulative sequences. Cumulative sequence removes all pad tokens from "
                    "original inputs, and stride all other tokens within seq_len dimension. This is very useful to "
                    "decrease memory usages and speed up training. Flash attention is mandatory during the model forward."
        }
    )
    pooling_strategy: str = field(
        default=None,
        metadata={
            "help": "Pooling strategy. Choose between mean/cls/lasttoken."
        },
    )
    score_function: str = field(
        default="cos_sim",
        metadata={
            "help": "Pooling strategy. Choose between dot/cos_sim."
        },
    )
    normalize: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to l2 normalize the representation. This feature is controlled by `score_function`."
                    "`score_function==dot`: `normalize=False`."
                    "`score_function==cos_sim`: `normalize=True`."
        },
    )

    # Indivisual settings
    model_name_or_path_qry: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
                    "Will be overriden if `model_name_or_path` is set."
        },
    )
    model_name_or_path_psg: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
                    "Will be overriden if `model_name_or_path` is set."
        },
    )
    pooling_strategy_qry: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pooling strategy of query model. Choose between mean/cls/lasttoken. Will be overriden by `pooling_strategy`"
        },
    )
    pooling_strategy_psg: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pooling strategy of passage model. Choose between mean/cls/lasttoken. Will be overriden by `pooling_strategy`"
        },
    )
    projection_in_dim_qry: Optional[int] = field(
        default=None,
        metadata={"help": "MLP Fan-in. Will be overrided if `projection_in_dim` is set."}
    )
    projection_out_dim_qry: Optional[int] = field(
        default=None,
        metadata={"help": "MLP Fan-out. Will be overrided if `projection_out_dim` is set."}
    )
    projection_in_dim_psg: Optional[int] = field(
        default=None,
        metadata={"help": "MLP Fan-in. Will be overrided if `projection_in_dim` is set."}
    )
    projection_out_dim_psg: Optional[int] = field(
        default=None,
        metadata={"help": "MLP Fan-out. Will be overrided if `projection_out_dim` is set."}
    )

    # *** Hybrid Model ***
    hybrid_use_dense_vector: bool = field(
        default=False,
        metadata={"help": "Train & Encode using dense vector."}
    )
    hybrid_use_sparse_vector: bool = field(
        default=False,
        metadata={"help": "Train & Encode using sparse vector."}
    )
    sparse_use_relu: bool = field(
        default=False,
        metadata={"help": "Whether to use relu for pooling sparse vectors"}
    )
    sparse_use_log_saturation: bool = field(
        default=False,
        metadata={"help": "Whether to use log saturation for pooling sparse vectors"}
    )
    sparse_use_logsumexp: bool = field(
        default=False,
        metadata={"help": "Whether to use log sum exp to estimite max function for pooling sparse vectors"}
    )
    sparse_use_mean_aggregation: bool = field(
        default=False,
        metadata={"help": "Whether to use mean function for pooling sparse vectors. If False, using max aggregation."}
    )
    sparse_min_tokens_to_keep: int = field(
        default=8,
        metadata={"help": "Min tokens to keep for top-p / top-k sampling."}
    )
    sparse_top_p_qry: float = field(
        default=1.0,
        metadata={"help": "Top-p of nucleus sampling, range [0, 1]."}
    )
    sparse_top_p_psg: float = field(
        default=1.0,
        metadata={"help": "Top-p of nucleus sampling, range [0, 1]."}
    )
    sparse_top_k_qry: int = field(
        default=1024,
        metadata={
            "help": "Top-k of nucleus sampling, 0 to disable."
                    "Hard limit of query topk. Avoid error: https://github.com/castorini/anserini/issues/745"
        }
    )
    sparse_top_k_psg: int = field(
        default=0,
        metadata={"help": "Top-k of nucleus sampling, 0 to disable."}
    )
    sparse_use_adaptive_top_k: bool = field(
        default=False,
        metadata={"help": "Whether to set top-k based on number of unique tokens of inputs."}
    )
    sparse_expansion_ratio_qry: float = field(
        default=2.0,
        metadata={"help": "Query Expansion ratio of adaptive top-k sampling for sparse embedding pooling."}
    )
    sparse_expansion_ratio_psg: float = field(
        default=2.0,
        metadata={"help": "Passage Expansion ratio of adaptive top-k sampling for sparse embedding pooling."}
    )
    sparse_pool_from_input_ids: bool = field(
        default=False,
        metadata={"help": "Whether to pool the embedding only from token appear in the input ids."}
    )
    sparse_select_top_rank_basis: bool = field(
        default=False,
        metadata={
            "help": "Whether to pool the sparse embedding by choosing the embedding matrix basis (token id) from top ranked "
                    "bitwise products. Assume we have query logits q = [q1, q2, ..., qn] and p = [p1, p2, ..., pn], then "
                    "the sparsity is chosen with following training procedure: "
                    "1) Training: Take the `top-k scores` of q*p bitwise multiplication `top-k([q1p1, q2p2, ..., qnpn])` as "
                    "inner-production similarities of sparsified query-passage emebddings. "
                    "2) Inferencing: Take the `top-k` logits of q (or p) from both ascend and descend sides. "
                    " - Descend side: preserve positive values. "
                    " - Ascend side: preserve negative values. Add `neg_` prefix to the token."
        }
    )
    sparse_basis_top_k: int = field(
        default=128,
        metadata={"help": "Top-k of top-rank basis."}
    )
    sparse_basis_use_largest: bool = field(
        default=True,
        metadata={
            "help": "Use Descend side when inferencing with embedding matrix basis (token id) of top ranked products."
        }
    )
    sparse_basis_use_smallest: bool = field(
        default=True,
        metadata={
            "help": "Use Ascend side when inferencing with embedding matrix basis (token id) of top ranked products."
        }
    )

    # *** ImbalancedEncoderModel ***
    # Query Model is a part of Passage Model.
    num_layers_for_query_model: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of layers for query model." 
                    "Default `None`: Using all layers."
                    "0: Only using the embeddings."
        }
    )

    def __post_init__(self):
        if self.score_function == "dot":
            self.normalize = False
        elif self.score_function == "cos_sim":
            self.normalize = True
        else:
            raise ValueError(f"The score function is {self.score_function}. This is not supported yet.")
        
        if self.model_name_or_path:
            self.model_name_or_path_qry = self.model_name_or_path
            self.model_name_or_path_psg = self.model_name_or_path
            if self.untie_encoder:
                _qry_model_path = os.path.join(self.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(self.model_name_or_path, 'passage_model')
                if os.path.exists(_qry_model_path) and os.path.exists(_psg_model_path):
                    self.model_name_or_path_qry = _qry_model_path
                    self.model_name_or_path_psg = _psg_model_path
        
        if self.pooling_strategy:
            self.pooling_strategy_qry = self.pooling_strategy
            self.pooling_strategy_psg = self.pooling_strategy
        
        if self.projection_in_dim:
            self.projection_in_dim_qry = self.projection_in_dim
            self.projection_in_dim_psg = self.projection_in_dim
        
        if self.projection_out_dim:
            self.projection_out_dim_qry = self.projection_out_dim
            self.projection_out_dim_psg = self.projection_out_dim


@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    dataloader_drop_last: bool = field(
        default=True, 
        metadata={
            "help": "Drop the last incomplete batch if it is not divisible by the batch size."
                    "This is mendatory for embedding training."
        }
    )

    # Model Implementation Related
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature scale for clloss."}
    )
    clloss_coef: float = field(
        default=1.0,
        metadata={"help": "Scale factor for clloss."}
    )
    distillation: bool = field(
        default=False,
        metadata={"help": "KL loss between Retriever query-passage scores and CrossEncoder scores."}
    )
    loss_reduction: str = field(
        default='mean', metadata={"help": "Loss reduction of CrossEntropy Loss. Choose among `mean`, `none`."}
    )
    negatives_x_device: bool = field(
        default=False, 
        metadata={
            "help": "Share negatives across global ranks. This is the `traditional` implemention of Cross Batch Negatives."
        }
    )
    loss_scaling_dro_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to config file of DRO optimized weights. The loss scale factor will be adjusted to "
            "`n_groups * group_weights`."
        },
    )
    dynamic_subbatch_forward: bool = field(
        default=False, 
        metadata={
            "help": "Whether to divide large batch into mini-subbatch in a forward process."
                    "[BGE M3-Embedding](https://arxiv.org/abs/2402.03216) utilize this technique"
                    "to save GPU memories of intermedite activations. This should be used with"
                    "`gradient_checkpoint` to task effect."
        }
    )

    ## Hybrid Model
    add_flops: bool = field(default=False)
    add_vector_norm: bool = field(default=False)
    norm_ord: int = field(default=1)
    q_norm_loss_factor: float = field(default=1.0)
    p_norm_loss_factor: float = field(default=1.0)

    # Trainer Related
    min_lr_ratio: float = field(default=0.0)
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    attn_implementation: str = field(
        default="flash_attention_2", metadata={"help": "Choose among `flash_attention_2`, `sdpa`, `eager`."}
    )
    logging_path: Optional[str] = field(
        default=None, metadata={"help": "Path for redirecting Transformers logs to local file."}
    )

    # Peft Config
    lora: bool = field(default=False, metadata={"help": "Use LoRA in Fine-tuning."})
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension (the \"rank\")."})
    lora_alpha: int = field(default=32, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.1, metadata={"help": "The dropout probability for Lora layers."})

    # *** GradCache ***
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    grad_cache_ac: bool = field(default=False, metadata={"help": "Use gradient cache + activation checkpointing"})
    gc_q_chunk_size: int = field(default=32)
    gc_p_chunk_size: int = field(default=4)
    no_sync_except_last: bool = field(
        default=False, 
        metadata={
            "help": "Whether to disable grad sync for GradCache accumulation backwards."
                    "This helps reduces the communication overhead of accumulated backwards."
                    "But it can induce more GPU memory usage for FSDP."
                    "Also, Deepspeed is not compatiable with this behavior."
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if self.resume_from_checkpoint is not None:
            if self.resume_from_checkpoint.lower() in ["false", 'f']:
                self.resume_from_checkpoint = None
            elif self.resume_from_checkpoint.lower() in ["true", 't']:
                self.resume_from_checkpoint = True
