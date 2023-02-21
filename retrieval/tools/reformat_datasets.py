#!/usr/bin/env python
# coding=utf-8

import os
import sys
sys.path.insert(1, os.getcwd())
import logging
import argparse
import jsonlines
from datasets import load_from_disk


def generate_global_dic(raw_dataset):
    global_dic = {k:[] for k in raw_dataset.keys()}
    # Main loop
    for k in raw_dataset.keys():
        idx = 0
        for idx, elem in enumerate(raw_dataset[k]):
            #print(f"raw_dataset[{k}] == {raw_dataset[k]}")
            doc_list = []
            for lbl, index in zip(elem["answerability_labels"], elem["indices"]):
                label = 1 if lbl else 0
                doc_list.append({"psg_key": index, "psg_lbl": label, "sent_lbl": []})
            global_dic[k].append({"idx": idx, "qus": elem["question"], "can_lst": doc_list})
    return global_dic




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
        "--output_dir",
        type=str,
        default=None,
        help="The directory where to save the reformated dataset.",
    )

    args = parser.parse_args()
    logger.info(f"Loading labeled datasets ...")
    raw_dataset = load_from_disk(args.dataset_path)
    logger.info(f"Reformating datasets ...")
    global_dic = generate_global_dic(raw_dataset) 
    logger.info(f"Saving datasets to disk...")
    os.makedirs(args.output_dir, exist_ok=True)
    for k in global_dic.keys():
        with jsonlines.open(os.path.join(args.output_dir, k + '.jsonl'), 'w') as writer:
            writer.write_all(global_dic[k])

    


    

    



if __name__ == "__main__":
    main()