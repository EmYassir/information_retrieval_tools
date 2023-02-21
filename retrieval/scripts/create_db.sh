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
export ROOT_DIR=/home/yassir/qa_gan_project/test_code
export DATA_PATH=/media/data/yassir/datasets/tokenized/wikipedia_split_tokenized_truncated_with_title
export OUTPUT_DIR=/media/data/yassir/datasets
export FILENAME=database_bis


# Settings
export BATCH_SIZE=500
export SEED=56

#torchrun --nproc_per_node=$N_GPU_NODE \
#python -m torch.distributed.launch --nproc_per_node=$N_GPU_NODE  \
#    --nnodes=$N_NODES \
#    --node_rank $NODE_RANK \
#    --master_addr $MASTER_ADDR \
#    --master_port $MASTER_PORT \
python   ${ROOT_DIR}/create_db_mp.py --n_gpu ${N_GPU_NODE} \
        --tokenized_dataset_path ${DATA_PATH} \
        --per_device_batch_size ${BATCH_SIZE} \
        --output_dir ${OUTPUT_DIR} \
        --output_file_name ${FILENAME} \
        --seed ${SEED} 
