#!/usr/bin/env python
# coding=utf-8

import os
import sys
sys.path.insert(1, os.getcwd())
import logging
import jsonlines
import csv
import argparse
from shutil import rmtree
from tqdm.auto import tqdm



def main():
    logger = logging.getLogger(__name__)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Generate the global wikipedia corpus")
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
        help="The path where to save the corpus.",
    )
    args = parser.parse_args()
    logger.info(f"Loading the csv file and writing to disk ...")
    counter = 0
    with open(args.dataset_path, "r") as fr:
        rd = csv.reader(fr, delimiter="\t", quotechar='"')
        with jsonlines.open(args.output_dir, 'w') as fw:
            for row in tqdm(rd):
                try:
                    idx, text = str(int(row[0]) - 1), row[1]
                except ValueError:
                    continue
                fw.write({"psg_key": idx, "sentences": [text]})
                counter += 1
    logger.info(f"Wrote {counter} lines to disk ...")

if __name__ == "__main__":
    main()