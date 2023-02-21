
import os
import sys
import argparse
from datasets import load_dataset
from transformers import (
    DefaultDataCollator,
    DPRContextEncoderTokenizer,
)

import logging

def add_sep(example, col_name):
    example[col_name] = example[col_name] + '[SEP]'
    print(example)
    return example

def concat(example, col_names, new_col_name):
    example[new_col_name] = example[col_names[0]] + example[col_names[1]]
    print(example)
    return example

def tokenize_function(examples, text_column_name):
    processed = list(map(lambda(s1, s2): (s1 if s1 else "") + " [SEP] " + (s2 if s2 else ""), zip(examples[text_column_name[0]], examples[text_column_name[1]])))
    return tokenizer(processed, padding=True, truncation=True)



if __name__ == "__main__":
    # Logging
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # Parser
    parser = argparse.ArgumentParser(
        description="Parallel tokenizer"
    )
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--num_proc", type=int, default=2)
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        raise ValueError('"output_dir" should be a valid directory.')

    tokenizer_kwargs = {"use_fast": True}
    logger.info(f"## Loading pre-trained tokenizer...")
    tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", **tokenizer_kwargs)

    logger.info(f"********* Loading datafiles...")
    data_files = {}
    data_files["train"] = args.dataset_path
    ext = args.dataset_path.split(".")[-1]
    if ext == "txt":
        ext = "text"
    elif ext == "tsv":
        ext = "csv"
    datasets = load_dataset(ext, data_files=data_files,  delimiter="\t", ignore_verifications=True)

    column_names = datasets["train"].column_names
    text_column_names = ["title", "text"]

    print(">>>>>>>>> [START] Tokenizing datasets ...")
    tokenized_datasets = datasets.map(
        tokenize_function,
        fn_kwargs={"text_column_names": text_column_names}, 
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        load_from_cache_file=True
    )
    logger.info("<<<<<<<<< [END] Tokenizing datasets ...")
    logger.info(f"Saving tokenized datasets to the directory \'{args.output_dir}\' ...")
    tokenized_datasets.save_to_disk(args.output_dir)
    print("DONE")
