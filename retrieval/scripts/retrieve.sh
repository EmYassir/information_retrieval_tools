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
#export DATASET="WebQuestions"
export DATASET_PATH=/media/data/yassir/datasets/qa/${DATASET}
export CORPUS_PATH=/media/data/yassir/databases/psgs_w100.tsv
export INDEX_PATH=/media/data/yassir/databases/database_bis.faiss
export OUTPUT_DIR=/media/data/yassir/output/datasets/${DATASET}

# Settings
export WORLD_SIZE=8
export NUM_PROC=64
export BATCH_SIZE=512
export RETRIEVER="dpr"
export K_PASSAGES=100

python ${ROOT_DIR}/tools/RetrieveDocsOpenQA.py \
    --dataset ${DATASET} \
    --dataset_path ${DATASET_PATH} \
    --retriever ${RETRIEVER} \
    --corpus_path ${CORPUS_PATH} \
    --cache_dir ${HF_HOME} \
    --output_dir ${OUTPUT_DIR} \
    --index_path ${INDEX_PATH} \
    --world_size ${WORLD_SIZE} \
    --k_passages ${K_PASSAGES} \
    --num_procs ${NUM_PROC} \
    --batch_size ${BATCH_SIZE}
   
#--ignore_cache_dirs
