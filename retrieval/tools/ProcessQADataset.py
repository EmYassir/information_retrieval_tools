import sys
import os
import argparse
from concurrent.futures import process
sys.path.insert(1, os.getcwd())

from src.utilities.DatasetProcessors.QuasarTProcessor import  QuasarTProcessor
QUASAR_T =  "QuasarT"
SEARCH_QA = "SearchQA"
DATASET_CHOICES = [QUASAR_T, SEARCH_QA]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extracts the top  100 passages for each question. Returns answerable passages and Reader scores."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=DATASET_CHOICES,
        help="OpenQA Dataset to retrieve documents for.",
    )
    parser.add_argument(
        "--dataset_path", required=True, type=str, help="Path to the chosen dataset"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Directory to output dataset with documents.",
    )
    return parser.parse_args()
def get_dataset_processor(args):
    if args.dataset == QUASAR_T:
        dataset_processor = QuasarTProcessor(
            args.dataset_path
        )
    elif args.dataset == SEARCH_QA:
        raise Exception("Unimplemented")
    return dataset_processor

if __name__ == "__main__":
    args = parse_args()
    processor = get_dataset_processor(args)
    processor.get().save_to_disk(args.output_path)
    