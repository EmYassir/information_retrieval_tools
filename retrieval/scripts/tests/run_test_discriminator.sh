#!/bin/bash

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

# Paths and options
export ROOT_DIR=~/qa_gan_project
export MAIN=$ROOT_DIR/tools/test_generator.py
export OPTION=discriminator
export CONFIG_PATH=$ROOT_DIR/cfg/gen_default_config.json
export OUTPUT_DIR=/media/data/yassir/output/test_generator
export DEVICE=0



echo "************** Starting the test **************"
python $MAIN --option $OPTION \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE




