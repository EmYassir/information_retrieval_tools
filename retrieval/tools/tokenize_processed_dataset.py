import sys
import os
import argparse
from transformers import BertTokenizerFast

sys.path.insert(1, os.getcwd())
from src.model.utilities.trainer_utils import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the dataset to use after saving to disk with a DatasetProcessor.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to store the tokenized dataset.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=350,
        help=("Set the max length to truncate to."),
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenized_dataset = prepare_dataset(args, tokenizer)
    tokenized_dataset.save_to_disk(args.output_path)
    print(tokenized_dataset)
