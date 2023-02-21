#!/usr/bin/env python
# coding=utf-8

import os
import sys
sys.path.insert(1, os.getcwd())
import logging

import re
import argparse
from shutil import rmtree

from datasets import load_from_disk, DatasetDict
from tqdm.auto import tqdm



def remove_dir(dir_path, logger):
    try:
        rmtree(dir_path)
    except OSError as e:
        logger.warn(f"Warning: {dir_path} : {e.strerror}")



def label_datasets(args, raw_datasets, logger):
    #nonlocal logger
    # Answerability labeling for each document
    def label_answerable_examples(example):
        def locate_answer(w):
            return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search

        compile_func = locate_answer(
            "|".join([re.escape(answer) for answer in example["answer"]])
        )
        answerability_labels = list(
            map(bool, (map(compile_func, example["documents"])))
        )
        return {"answerability_labels": answerability_labels}
    


    logger.info(f"===>>> Labeling the dataset...")
    
    # If test set doesn't contain answers, ignore it
    if "answer" not in raw_datasets["test"].column_names:
        raw_datasets = DatasetDict(
            {"train": raw_datasets["train"], "dev": raw_datasets["dev"]}
        )
    answerable_datasets = raw_datasets.map(
        label_answerable_examples,
        num_proc=args.num_processes,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=['documents', 'titles'],
        desc="Labeling each document as having an answer or not",
    )
    

    logger.info(f"===>>> Filtering examples with no positive document...")
    # Keep only questions that have at least one answerable document
    answerable_datasets = answerable_datasets.filter(
        lambda example: any(example["answerability_labels"]),
        num_proc=args.num_processes,
        desc="Filtering the datasets to have at least one question",
    )

    return answerable_datasets




def main():
    logger = logging.getLogger(__name__)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The path to the raw dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The path where to save the processed dataset.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Whether or not use the cache.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=16,
        help="Number of processes used for data preprocessing.",
    )
    args = parser.parse_args()
    logger.info(f"Loading datasets ...")
    raw_dataset = load_from_disk(args.dataset_path)
    logger.info(f"Labeling and extending datasets ...")

    ans_ds = label_datasets(args, raw_dataset, logger)
    print(f"Length of answerable training dataset: {len(ans_ds['train'])}")
    
    logger.info(f"Saving labeled datasets to disk ...")
    os.makedirs(args.output_path, exist_ok=True)
    ans_ds.save_to_disk(args.output_path)

    



if __name__ == "__main__":
    main()