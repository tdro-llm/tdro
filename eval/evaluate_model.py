import os
import sys
import time
import json
import numpy as np
from pathlib import Path
import logging
logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if (int(os.getenv("RANK", -1)) in [0, -1]) else logging.WARN,
    force=True,
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
torch.set_float32_matmul_precision('high')

# Hacking with Fused Ops
from tdro.utils.monkey_patch import hacking_fused_rms_norm
hacking_fused_rms_norm()

import mteb
from transformers import HfArgumentParser, set_seed
from eval_arguments import EvalArguments
from modeling_utils import ExactSearchModel, get_mteb_prompt
from load_utils.beir_dataloader import load_beir_data, BEIR_TASKS
from load_utils.miracl_dataloader import load_miracl_data

from tdro.retriever.faiss_search import FlatIPFaissSearch


TASK_DICT = {
    "retrieval": BEIR_TASKS,
}

def main(args: EvalArguments, model: nn.Module):
    # Hacking with customized search function
    # mteb/evaluation/evaluators/RetrievalEvaluator.py#478
    hackable_meta_of_mteb = mteb.model_meta.ModelMeta(
                                loader=None,  # type: ignore
                                name="bm25s",   # Hacking point: name `bm25s` with call customized .search() func
                                                # Thus need not to modify the MTEB code
                                languages=["eng_Latn"],
                                open_source=True,
                                revision=Path(args.model_name_or_path).name,     # Change to exp name !!
                                release_date="2024-01-01",  # initial commit of hf model.
                            )

    if args.task_type is not None:
        if args.task_name is not None:
            logger.warning(f"Task type {args.task_type} is set. This will override the settings of task name!!!")
        task_names = TASK_DICT[args.task_type]
    else:
        task_names = [args.task_name]
    logger.info(f"Running evaluation on tasks: {task_names}")
    
    retriever = FlatIPFaissSearch(model, args.batch_size, use_multiple_gpu=True)
    retriever.mteb_model_meta = hackable_meta_of_mteb
    
    tasks = mteb.get_tasks(tasks=task_names, languages=args.lang)
    for task_cls in tasks:
        task_name: str = task_cls.metadata.name
        task_type: str = task_cls.metadata.type

        model.query_prompt, model.corpus_prompt = "", ""
        if args.add_prompt:
            model.query_prompt, model.corpus_prompt = get_mteb_prompt(task_name=task_name, task_type=task_type, prompt_type=args.prompt_type)
            
            logger.info(f'Set query prompt: {json.dumps(model.query_prompt, ensure_ascii=False)}')
            logger.info(f'Set corpus prompt: {json.dumps(model.corpus_prompt, ensure_ascii=False)}')
        
        # Override disable l2 normalize for classification tasks, as it achieves slightly better results
        if task_type == 'Classification':
            logger.info('Set l2_normalize to False for classification task')
            model.encoding_kwargs['normalize'] = False
        else:
            if args.score_function == "cos_sim":
                model.encoding_kwargs['normalize'] = True
                logger.info('Set l2_normalize to {}'.format(model.encoding_kwargs['normalize']))

        if task_name == 'MSMARCO':  # MSMARCO uses dev
            eval_splits = ['dev']
        elif "test" in task_cls.metadata.eval_splits:   # If this task has a `test` set, only use this set
            eval_splits = ["test"]
        else:                       # Other conditions, use original settings
            eval_splits = task_cls.metadata.eval_splits
        
        # Load BeIR data locally
        if task_name in BEIR_TASKS:
            logger.info(f"Loading {task_name} data")
            task_cls.corpus, task_cls.queries, task_cls.relevant_docs = load_beir_data(task_name, splits=eval_splits)
            task_cls.data_loaded = True
            logger.info(f"Dataset {task_cls} loaded. Splits: {task_cls.metadata_dict['eval_splits']}")
        
        # Load MIRACL data locally
        if task_name == "MIRACLRetrieval":
            logger.info(f"Loading {task_name} data")
            task_cls.corpus, task_cls.queries, task_cls.relevant_docs = load_miracl_data(langs=task_cls.hf_subsets, splits=eval_splits)
            task_cls.data_loaded = True
            logger.info(f"Dataset {task_cls} loaded. Splits: {task_cls.metadata_dict['eval_splits']}")
        
        logger.info('=== Running evaluation for Task: {}, Type: {} , Splits: {} === \n'.format(task_name, task_type, eval_splits))

        sub_eval = mteb.MTEB(tasks=[task_cls])
        sub_eval.run(
            retriever, 
            verbosity=2, 
            output_folder=args.output_dir, 
            eval_splits=eval_splits, 
            overwrite_results=args.overwrite_results, 
            save_predictions=args.save_predictions,
            top_k=args.top_k,
            previous_results=args.previous_results,
        )

        del sub_eval
        del task_cls

    logger.info("--DONE--")


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
