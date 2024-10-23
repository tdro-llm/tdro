# Evaluation
This folder holds evaluation guidelines for BeIR, MIRACL and MKQA. Our evaluation pipeline is developed based on [Distributed RPC Framework](https://pytorch.org/docs/stable/rpc.html), which akins the pattern of [one producer - multiple consumers (workers)](https://en.wikipedia.org/wiki/Producer–consumer_problem) and naturally supports multi-node, multi-GPU encoding.

## Data Preparation
Please refer to scripts below for downloading the evaluation datasets.

**BeIR**: [`download_beir.sh`](load_utils/download_beir.sh)

**MIRACL**: [`download_miracl.sh`](load_utils/download_miracl.sh)

**MKQA**: [`download_mkqa.sh`](load_utils/download_mkqa.sh)

**Note**: BeIR, MIRACL are also hosted by [MTEB](https://github.com/embeddings-benchmark/mteb) with online Huggingface Datasets. You can also choose to load them online, bypassing the data loading blocks in [`evaluate_model.py#97`](evaluate_model.py#97). However, in our experiments, loading from local files is always the fastest and most stable way.

## Evaluation Scripts
First, please set some model arguments below:

```bash
# Global Model Arguments
MODEL_KWARGS=""
MODEL_KWARGS+=" --model_type EncoderModel "     # Only support EncoderModel for now
MODEL_KWARGS+=" --pooling_strategy lasttoken "  # Last token (</eos>) pooling. Make sure tokenizer appends a </eos> token
MODEL_KWARGS+=" --score_function cos_sim "      # Cosine similarity
MODEL_KWARGS+=" --q_max_len 128 "               # Query max length
MODEL_KWARGS+=" --p_max_len 512 "               # Passage max length
MODEL_KWARGS+=" --bf16 "                        # Bfloat16 training / inferencing (Mix-precision w/ auto-cast)
MODEL_KWARGS+=" --add_prompt "                  # Whether to add prompt in front of the queries
MODEL_KWARGS+=" --prompt_type e5 "              # Here we follow the prompt settings of Mistral-E5

# Set General Model Arguments
export MODEL_KWARGS=$MODEL_KWARGS
```

Assume the retriever (folder name `TRAIL_NAME`) is located in `tdro/results/$TRAIL_NAME`. Please execute the following commands:

```bash
bash test_beir.sh $TRAIL_NAME
bash test_miracl.sh $TRAIL_NAME
bash test_mkqa.sh $TRAIL_NAME
```

## Acknowledgement
Our evaluation pipeline is developed based on [BeIR](https://github.com/beir-cellar/beir), [MIRACL](https://github.com/project-miracl/miracl), [MKQA](https://huggingface.co/datasets/apple/mkqa), [MTEB](https://github.com/embeddings-benchmark/mteb).
