import argparse
import os
import sys
from pprint import pprint
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)

sys.path.insert(1, os.getcwd())
from src.model.utilities.trainer_utils import (
    prepare_dataset,
    get_collate_fn,
    get_eval_metrics,
)
from src.config.gan_config import FullGANConfig
from src.model.gan_modeling import FullGANModel


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
        "--is_tokenized",
        action="store_true",
        help="If set to true, the dataset is assumed to be tokenized and tokenization is not performed",
    )
    parser.add_argument(
        "--train_dataset",
        action="store_true",
        help="If set to true, runs evaluation on the train dataset.",
    )
    parser.add_argument(
        "--test_dataset",
        action="store_true",
        help="If set to true, runs evaluation on the test dataset.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the training configuration with gernator, rank_discriminator, and answerability_discriminator configurations.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        required=True,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--ans_discriminator_weight",
        type=int,
        default=0.25,
        help=(
            "Weight to assign to answerability discriminator's reward for generator loss."
        ),
    )
    parser.add_argument(
        "--regularizer_weight",
        type=int,
        default=1,
        help=(
            "Weight to assign to regularizing factor in the reward for generator loss."
        ),
    )
    parser.add_argument(
        "--hits_list",
        type=list,
        default=[1, 3, 5, 10, 20, 30, 50],
        help=("List of N for Hits@N evaluation."),
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    # Ensure that the hits list is a list of integers
    if args.hits_list is not None:
        args.hit_list = [int(hit) for hit in args.hits_list]

    # Prepare the datasets
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if args.is_tokenized:
        tokenized_datasets = load_from_disk(args.dataset_path)
    else:
        tokenized_datasets = prepare_dataset(
            args, BertTokenizerFast.from_pretrained("bert-base-uncased")
        )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["dev"]
    test_dataset = tokenized_datasets["test"]
    print("EVAL DATASET LENGTH: ", len(eval_dataset))
    print("Train DATASET LENGTH: ", len(train_dataset))
    print("Test DATASET LENGTH: ", len(test_dataset))

    # Initialize the models
    gen_device = torch.device("cuda:0")
    dis_device = torch.device("cuda:1")
    ans_device = torch.device("cuda:1")
    model_name = "bert-base-uncased"
    generator = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(gen_device)
    discriminator = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(dis_device)
    ans_discriminator = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(ans_device)

    # Load models
    checkpoint = torch.load(args.resume_from_checkpoint)
    generator.load_state_dict(checkpoint["gen_model_state_dict"])
    discriminator.load_state_dict(checkpoint["dis_model_state_dict"])
    ans_discriminator.load_state_dict(checkpoint["ans_model_state_dict"])

    # Get collate function:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    data_collator = get_collate_fn(DataCollatorWithPadding(tokenizer))

    # Create eval dataloader, perform evaluation, and print the metrics
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    print("EVAL DATALOADER LENGTH: ", len(eval_dataloader))
    eval_metrics = get_eval_metrics(
        args,
        generator,
        discriminator,
        ans_discriminator,
        eval_dataloader,
        gen_device,
        dis_device,
        ans_device,
    )
    print("================ EVAL METRICS ==================")
    pprint(eval_metrics, sort_dicts=False)

    # Create test dataset dataloader, perform evaluation and print the metrics
    if args.test_dataset:
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )
        test_metrics = get_eval_metrics(
            args,
            generator,
            discriminator,
            ans_discriminator,
            test_dataloader,
            gen_device,
            dis_device,
            ans_device,
        )
        print("================ TEST METRICS ==================")
        pprint(test_metrics, sort_dicts=False)

    # Create the train dataloader, perform evaluation and print the metrics
    if args.train_dataset:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )

        train_metrics = get_eval_metrics(
            args,
            generator,
            discriminator,
            ans_discriminator,
            train_dataloader,
            gen_device,
            dis_device,
            ans_device,
        )
        print("================ TRAIN METRICS ==================")
        pprint(train_metrics, sort_dicts=False)
