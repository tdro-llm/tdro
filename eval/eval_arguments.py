import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

from tdro.finetune.arguments import ModelArguments

import torch
logger = logging.getLogger(__name__)

@dataclass
class EvalArguments(ModelArguments):
    """
    Eval Arguments for benchmarks.
    """
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for evaluation."}
    )
    model_type: Optional[str] = field(
        default="EncoderModel",
        metadata={
            "help": "The model archeticture used in training."
                    "Choose among ['EncoderModel']."
        },
    )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Folder to save the output results."}
    )
    task_name: Optional[str] = field(
        default=None, metadata={"help": "Single task name to evaluate."}
    )
    task_type: Optional[str] = field(
        default=None, metadata={"help": "Pre-defined task type for evaluting multiple tasks."}
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size for encoding."}
    )
    add_prompt: bool = field(
        default=False, metadata={"help": "Whether to add query prompt."}
    )
    prompt_type: str = field(
        default="e5", metadata={"help": "Type of query prompt, choose among e5, instructor, bge, e5_ori."}
    )
    eval_all_langs: bool = field(
        default=False, metadata={"help": "Whether to use evaluate all languages."}
    )
    lang: str = field(
        default='en', metadata={"help": "Languages to evaluate. Please use `,` to seperate."}
    )
    overwrite_results: bool = field(
        default=False, metadata={"help": "Whether to override the existing results."}
    )
    save_predictions: bool = field(
        default=False, metadata={
            "help": "Whether to save preds. MTEB will save the json predictions"
                    "to `(output_folder)/(self.metadata.name)_(hf_subset)_predictions.json`"
        }
    )
    top_k: int = field(
        default=1000, metadata={"help": "The top-k threshold to retrieve or rerank."}
    )
    previous_results: Optional[str] = field(
        default=None, metadata={"help": "The json file path to previous stage retrieval results. The default `RetrievalEvaluator` will load it and then reranking it with `top_k`."}
    )
    sigmoid_normalize: bool = field(
        default=False,
        metadata={
            "help": "Whether to normalize the reranker output to range (0, 1) with `nn.Sigmoid()`."
        },
    )

    # Model args
    q_max_len: int = field(
        default=128, metadata={"help": "Query maxlen."}
    )
    p_max_len: int = field(
        default=512, metadata={"help": "Passage maxlen."}
    )
    sep: str = field(
        default=" ", metadata={"help": "Separator between title and text."}
    )
    max_length: int = field(
        default=512, metadata={"help": "Reranker maxlen."}
    )
    padding: Union[bool, str] = field(
        default=True,
    )
    pad_to_multiple_of: int = field(
        default=8,
    )
    bf16: bool = field(
        default=False, metadata={"help": "Bfloat16."}
    )
    fp16: bool = field(
        default=False, metadata={"help": "Float16."}
    )
    seed: int = field(
        default=42, metadata={"help": "Seed."}
    )
    sep: str = field(
        default=' ',
        metadata={
            "help": "Separator between title and passage."
        },
    )
    attn_implementation: str = field(
        default="flash_attention_2", metadata={"help": "Choose among `flash_attention_2`, `sdpa`, `eager`."}
    )
    torch_compile: bool = field(
        default=False, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )

    # RPC Related
    local_rank: int = field(
        default=-1, metadata={"help": "Local rank initilized by `LOCAL_RANK`."}
    )
    rank: int = field(
        default=-1, metadata={"help": "Global rank initilized by `RANK`."}
    )
    world_size: int = field(
        default=0, metadata={"help": "World size initilized by `WORLD_SIZE`."}
    )
    master_addr: str = field(
        default="127.0.0.1", metadata={"help": "Master address."}
    )
    master_port: int = field(
        default=12345, metadata={"help": "Master port."}
    )
    debug: bool = field(
        default=False, metadata={"help": "Debug encoding function"}
    )

    def __post_init__(self):
        super().__post_init__()
        
        # Init dist args
        if local_rank := os.getenv("LOCAL_RANK", None):
            self.local_rank = int(local_rank)
        if rank := os.getenv("RANK", None):
            self.rank = int(rank)
        if world_size := os.getenv("WORLD_SIZE", None):
            self.world_size = int(world_size)
        if master_addr := os.getenv("MASTER_ADDR", None):
            self.master_addr = str(master_addr)
        if master_port := os.getenv("MASTER_PORT", None):
            self.master_port = str(master_port)

        # Parse langs args
        if self.eval_all_langs:
            self.lang = None
        else:
            if ',' in self.lang:
                self.lang = [i.strip() for i in self.lang.split(',')]
            else:
                if isinstance(self.lang, str):
                    self.lang = [self.lang]

        # Parse dtype
        self.dtype = None
        if self.bf16:
            self.dtype = torch.bfloat16
        elif self.fp16:
            self.dtype = torch.float16

        # # attn_implementation = "eager" should be reverted after Transformers 4.41
        # if self.model_type == "DenseModel" and self.attn_implementation != "eager":
        #     logger.warn("Setting attn_implementation to eager because current BERT model does not support flash attention")
        #     self.attn_implementation = "eager"
        