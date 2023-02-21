#!/bin/bash

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME


# Paths
export ROOT_DIR=/home/yassir/qa_gan_project/test_code
export DATA_PATH=/media/data/yassir/datasets/wikipedia_split/psgs_w100.tsv
export OUTPUT_DIR=/media/data/yassir/datasets/tokenized/wikipedia_split_tokenized_truncated_with_title
export NUM_PROC=65

python ${ROOT_DIR}/tokenize_dataset.py --dataset_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --num_proc ${NUM_PROC}
