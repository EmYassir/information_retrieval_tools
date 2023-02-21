#!/bin/bash

## General
export NODE_RANK=0
export LOCAL_RANK=0
export RANK=0
export N_NODES=1
export N_GPU_NODE=8
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=8
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))
export MASTER_ADDR=127.0.0.1


## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME


# Paths
export SHARDS=8
export ROOT_DIR=/home/yassir/qa_gan_project/test_code
export DATA_PATH=/media/data/yassir/datasets/tokenized/wikipedia_split_tokenized_truncated_with_title
export OUTPUT_DIR=/media/data/yassir/datasets/tokenized/wikipedia_split_tokenized_truncated_with_title_shard_


python   ${ROOT_DIR}/shard_dataset.py \
        --dataset_path ${DATA_PATH} \
        --output_prefix ${OUTPUT_DIR} \
        --n_shards ${SHARDS} 
