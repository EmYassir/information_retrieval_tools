import re
import sys
import os
import argparse
from transformers import (
    DPRReaderTokenizerFast,
    DPRReader,
    BatchEncoding,
    DataCollatorWithPadding,
    DPRReaderOutput,
)
from datasets import load_from_disk, DatasetDict, Dataset, Value
from tqdm import tqdm
import pandas as pd
import torch

sys.path.insert(1, os.getcwd())
from src.model.utilities.trainer_utils import (
    get_collate_fn,
    label_answerability,
)
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize dataset and set DPR relevance logits and DPR predictions."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the dataset to use after saving to disk with a DatasetProcessor.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Cache directory for data processing.",
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
        default=256,
        help=("Set the max length to truncate to."),
    )
    args = parser.parse_args()
    if args.cache_dir.endswith("/"):
        args.cache_dir = args.cache_dir[:-2]
    return args


def prepare_dataset(args, tokenizer):
    if args.cache_dir is not None and os.path.exists(args.cache_dir + "/DPR_out"):
        tokenized_datasets = load_from_disk(args.cache_dir + "/DPR_out")
    else:
        # Get the datasets: Loads a processed dataset from disk. Must be a dataset created by a DatasetProcessor
        raw_datasets = load_from_disk(args.dataset_path)

        # If test set doesn't contain answers, ignore it. This is the case for TriviaQA
        if "answer" not in raw_datasets["test"].column_names:
            raw_datasets = DatasetDict(
                {"train": raw_datasets["train"], "dev": raw_datasets["dev"]}
            )

        # Create Answerability Labels
        answerable_datasets = raw_datasets.map(
            label_answerability,
            num_proc=args.preprocessing_num_workers,
            desc="Labeling each document as having an answer or not",
        )

        # Tokenize the questions and documents
        def tokenize_function(example):
            # Prepend each document with the question and title
            return tokenizer(
                questions=example["question"],
                titles=example["titles"],
                texts=example["documents"],
                max_length=args.max_seq_len,
                truncation=True,
            )

        tokenized_datasets = answerable_datasets.map(
            tokenize_function,
            num_proc=args.preprocessing_num_workers,
            desc="Running tokenizer on every text in dataset",
        )

        # Prepare for running DPRReader on the entire dataset
        collator = get_collate_fn(
            DataCollatorWithPadding(tokenizer, return_tensors="pt")
        )
        tokenized_datasets.set_format(
            type=None, columns=["input_ids", "attention_mask", "answerability_labels"]
        )
        eval_dl = DataLoader(
            tokenized_datasets["dev"], batch_size=1, collate_fn=collator
        )
        device = torch.device("cuda:0")
        reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base").to(
            device
        )

        # Run DPRReader on dataset and collect its outputs
        dpr_input = []
        start_logits = []
        end_logits = []
        relevance_logits = []
        for batch in tqdm(eval_dl, desc="Running reader on DS"):
            with torch.no_grad():
                reader_input = BatchEncoding(
                    {
                        "input_ids": batch["input_ids"][0].to(device),
                        "attention_mask": batch["attention_mask"][0].to(device),
                    }
                )
                out = reader(**reader_input)

                dpr_input.append(reader_input.to("cpu"))
                start_logits.append(out["start_logits"].tolist())
                end_logits.append(out["end_logits"].tolist())
                relevance_logits.append(out["relevance_logits"].tolist())

        # Add the aggregated results back into the dataset
        tokenized_datasets.set_format(type=None, output_all_columns=True)
        tokenized_datasets["dev"] = tokenized_datasets["dev"].add_column(
            "start_logits", start_logits
        )
        tokenized_datasets["dev"] = tokenized_datasets["dev"].add_column(
            "end_logits", end_logits
        )
        tokenized_datasets["dev"] = tokenized_datasets["dev"].add_column(
            "relevance_logits", relevance_logits
        )
        os.makedirs(args.cache_dir, exist_ok=True)
        tokenized_datasets.save_to_disk(args.cache_dir + "/DPR_out")

    # Extract the predicted answer from DPR Reader. 1 answer per document.
    def extract_best_answers(example):
        reader_output = DPRReaderOutput(
            start_logits=example["start_logits"],
            end_logits=example["end_logits"],
            relevance_logits=example["relevance_logits"],
        )
        reader_input = BatchEncoding(
            {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
        )
        best_spans = tokenizer.decode_best_spans(
            reader_input=reader_input,
            reader_output=reader_output,
            num_spans_per_passage=1,
            num_spans=100,
        )
        best_spans = sorted(best_spans, key=lambda x: x.doc_id)
        relevance_score = [row.relevance_score for row in best_spans]
        pred_answer = [row.text for row in best_spans]
        answer_doc_id = [row.doc_id for row in best_spans]
        return {
            "relevance_score": relevance_score,
            "pred_answer_span": pred_answer,
            "doc_id": answer_doc_id,
        }

    tokenized_datasets["dev"] = tokenized_datasets["dev"].map(
        extract_best_answers, num_proc=args.preprocessing_num_workers
    )

    # Check if DPR's predicted answer is correct;
    def verify_pred(example):
        answerability_labels = [
            any(list(map(lambda ans: pred.lower() == ans.lower(), example["answer"])))
            for pred in example["pred_answer_span"]
        ]
        return {"pred_answer": answerability_labels}

    tokenized_datasets["dev"] = tokenized_datasets["dev"].map(
        verify_pred, num_proc=args.preprocessing_num_workers
    )

    # Set the columns we wish to output. Modify this if there are columns you don't wish to discard
    out_column_names = {
        "dev": [
            "input_ids",
            "attention_mask",
            "answerability_labels",
            "relevance_logits",
            "pred_answer",
        ],
        "train": ["input_ids", "attention_mask", "answerability_labels"],
        "test": ["input_ids", "attention_mask", "answerability_labels"],
    }
    tokenized_datasets["dev"].set_format(
        type=None,
        columns=out_column_names["dev"],
    )
    tokenized_datasets["train"].set_format(
        type=None,
        columns=out_column_names["train"],
    )
    if "test" in tokenized_datasets:
        tokenized_datasets["test"].set_format(
            type=None,
            columns=out_column_names["test"],
        )

    # Flatten the datasets so that each row is a question and document instead of questions and k documents.
    def flatten_dataset(ds, out_column_names):
        final_dataset = {key: [] for key in out_column_names}
        final_dataset["id"] = []
        for step, item in enumerate(tqdm(ds, desc="Flattening dataset")):
            for key in out_column_names:
                final_dataset[key].extend(item[key])
            final_dataset["id"].extend([step] * len(item["input_ids"]))

        df = pd.DataFrame(final_dataset)
        return Dataset.from_pandas(df)

    train_ds = flatten_dataset(tokenized_datasets["train"], out_column_names["train"])
    train_ds.set_format(output_all_columns=True)
    eval_ds = flatten_dataset(tokenized_datasets["dev"], out_column_names["dev"])
    eval_ds.set_format(output_all_columns=True)
    return train_ds, eval_ds


if __name__ == "__main__":
    args = parse_args()
    tokenizer = DPRReaderTokenizerFast.from_pretrained(
        "facebook/dpr-reader-single-nq-base"
    )
    train_ds, eval_ds = prepare_dataset(args, tokenizer)
    train_ds.save_to_disk(args.output_path + "_train")
    eval_ds.save_to_disk(args.output_path + "_dev")
    print(train_ds)
    print(eval_ds)
