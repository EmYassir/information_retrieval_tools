#!/bin/bash
export NODE_RANK=0
export LOCAL_RANK=0
export RANK=0
export N_NODES=1
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))
export MASTER_ADDR=127.0.0.1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME


# Paths
export ROOT_DIR=/home/yassir/retrieval/
#export DATASET="NaturalQuestions"
#export DATASET="CuratedTREC"
export DATASET="WebQuestions"
export DATASET_PATH=/media/data/yassir/output/labeled_datasets/${DATASET}
export OUTPUT_DIR=/media/data/yassir/output/reformated_datasets/${DATASET}


python ${ROOT_DIR}/tools/reformat_datasets.py \
    --dataset_path ${DATASET_PATH} \
    --output_dir ${OUTPUT_DIR} 
