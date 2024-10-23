# Hard Negative Mining

## Sources from Sentence Transformers Training Data
Please first download coresponding raw data pairs below from [Sentence Transformers Training Data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), medmcqa pairs from [this link](https://huggingface.co/datasets/openlifescienceai/medmcqa). Then mine negatives with scripts below.

```bash
# Save all training datasets here
mkdir -p retrieval/original

# Dataset Choices
DATASETS=(agnews altlex amazon_review_2018 cnn_dailymail codesearchnet eli5_question_answer gooaq_pairs medmcqa searchQA_top5_snippets sentence-compression SimpleWiki squad_pairs stackexchange_duplicate_questions_title-body_title-body xsum yahoo_answers_title_answer)

# For example, mine negatives for `agnews`
DATASET_NAME=agnews
python build_train_hn_dp.py \
    --train_collection $PATH_TO/${DATASET_NAME}.jsonl.gz \
    --model_name_or_path BAAI/bge-base-en-v1.5 \
    --save_to retrieval/original/${DATASET_NAME}.jsonl
```

## Sources from BeIR Training Data
HotpotQA negative mining is made with source training queries and triples from [BeIR Training Data](https://github.com/beir-cellar/beir?tab=readme-ov-file#beers-available-datasets). Please first download and unzip [HotpotQA](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip). Then execute the following script.

Note that the BeIR dataset loading function (with packed queries, corpus and training qrel) is a little bit different from above codes. Please adjust the last line of `build_train_hn_dp.py` to `fire.Fire(mine_w_qrels)` for reading the data properly.

```bash
DATASET_NAME=hotpotqa
# Please using func mine_w_qrels
python build_train_hn_dp.py \
    --model_name_or_path BAAI/bge-base-en-v1.5 \
    --query_collection $PATH_TO/${DATASET_NAME}/queries.jsonl \
    --passage_collection $PATH_TO/${DATASET_NAME}/corpus.jsonl \
    --qrel_path $PATH_TO/${DATASET_NAME}/qrels/train.tsv \
    --save_to retrieval/original/${DATASET_NAME}.jsonl
```
