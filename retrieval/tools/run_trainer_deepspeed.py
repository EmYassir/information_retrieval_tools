#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.
import sys
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import re
import deepspeed

import datasets
from datasets import DatasetDict
import torch
from datasets import load_from_disk

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType

from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    MODEL_MAPPING,
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
    BertTokenizer,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

import torch.distributed as dist

sys.path.insert(1, os.getcwd())
from tools.trainer_utils import (
    compute_r_dis_loss,
    compute_gen_loss,
    get_collate_fn,
    get_generator_sample_mask,
    get_hits_dict,
)
from src.model.gan_modeling import FullGANModel
from src.config.gan_config import FullGANConfig


logger = logging.getLogger(__name__)
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=True,
        help="Path to the training configuration with gernator, rank_discriminator, and answerability_discriminator configurations.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the ???? Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--num_dis_epochs",
        type=int,
        default=3,
        help=("The number of discriminator epochs"),
    )
    parser.add_argument(
        "--num_gen_epochs",
        type=int,
        default=4,
        help=("The number of generator epochs"),
    )
    parser.add_argument(
        "--answerability_heuristic",
        action="store_true",
        help=(
            "Use heuristic for answerability instead of models. Utilizes BCELossWithLogits to compute answerability reward."
        ),
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
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args


def main(rank, world_size, train_dataset, eval_dataset):
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if rank == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # Use nested config.
    with open(args.config_path) as f:
        training_config = json.load(f)
    gan_config = FullGANConfig(
        generator_cfg=training_config["generator_cfg"],
        discriminator_cfg=training_config["discriminator_cfg"],
        ans_discriminator_cfg=training_config["ans_discriminator_cfg"],
    )
    gan = FullGANModel(gan_config)
    generator = gan.generator
    discriminator = gan.discriminator
    ans_discriminator = gan.ans_discriminator

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = get_collate_fn(DataCollatorWithPadding(tokenizer))

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    def init_optimizer(model):
        """Initializes optimizer and learning rate scheduler

        Args:
            model (torch.nn.module): Module to initialize optimizer and lr_scheduler for

        Returns:
            (torch.optim, lr_scheduler): Tuple of optimizer and lr_scheduler for the model
        """
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=args.learning_rate)

        # Scheduler and math around the number of training steps.
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        return optimizer, lr_scheduler

    # Initialize the optimizer for generator, discriminator, and answer discriminator
    gen_optimizer, gen_lr_scheduler = init_optimizer(generator)
    dis_optimizer, r_dis_lr_scheduler = init_optimizer(discriminator)
    ans_optimizer, a_dis_lr_scheduler = init_optimizer(ans_discriminator)

    (
        ans_discriminator_engine,
        ans_optimizer,
        ans_train_dataloader,
        a_dis_lr_scheduler,
    ) = deepspeed.initialize(
        args=args,
        model=ans_discriminator,
        model_parameters=ans_discriminator.parameters(),
        optimizer=ans_optimizer,
        lr_scheduler=a_dis_lr_scheduler,
        training_data=train_dataset,
    )
    (
        discriminator_engine,
        dis_optimizer,
        dis_train_dataloader,
        r_dis_lr_scheduler,
    ) = deepspeed.initialize(
        args=args,
        model=discriminator,
        model_parameters=discriminator.parameters(),
        optimizer=dis_optimizer,
        lr_scheduler=r_dis_lr_scheduler,
        training_data=train_dataset,
    )
    (
        generator_engine,
        gen_optimizer,
        gen_train_dataloader,
        gen_lr_scheduler,
    ) = deepspeed.initialize(
        args=args,
        model=generator,
        model_parameters=generator.parameters(),
        optimizer=gen_optimizer,
        lr_scheduler=gen_lr_scheduler,
        training_data=train_dataset,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=rank != 0)
    completed_steps = 0
    starting_epoch = 0
    starting_dis_epoch = 0
    starting_gen_epoch = 0
    skip_dis_phase = False

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            logger.log(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint)

            # Load models
            generator.load_state_dict(checkpoint["gen_model_state_dict"])
            discriminator.load_state_dict(checkpoint["dis_model_state_dict"])
            ans_discriminator.load_state_dict(checkpoint["ans_model_state_dict"])

            # Load optimizers
            gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
            dis_optimizer.load_state_dict(checkpoint["dis_optimizer_state_dict"])
            ans_optimizer.load_state_dict(checkpoint["ans_optimizer_state_dict"])

            # Determine step and epoch
            starting_epoch = checkpoint["epoch"]
            skip_dis_phase = checkpoint["current_phase"] == "generator"
            if skip_dis_phase:
                starting_gen_epoch = checkpoint["starting_epoch"]
            else:
                starting_dis_epoch = checkpoint["starting_epoch"]

    scaler = torch.cuda.amp.GradScaler()

    ans_loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(starting_epoch, args.num_train_epochs):
        generator.eval()
        discriminator.train()
        ans_discriminator.train()
        if args.resume_from_checkpoint and not epoch == starting_epoch:
            starting_dis_epoch = 0
            starting_gen_epoch = 0
        for d_epoch in range(starting_dis_epoch, args.num_dis_epochs):
            if args.with_tracking:
                total_rank_dis_loss = 0
                total_ans_dis_loss = 0
            for gen_batch, dis_batch, ans_batch in zip(
                gen_train_dataloader, dis_train_dataloader, ans_train_dataloader
            ):
                with torch.no_grad():
                    gen_scores = generator_engine(**gen_batch)["output_distribution"]
                dis_scores = discriminator_engine(**dis_batch)["output_distribution"]
                a_dis_scores = ans_discriminator_engine(**ans_batch)[
                    "output_distribution"
                ]

                true_dat_mask = batch["answerability_labels"]
                true_dis_scores = dis_scores[true_dat_mask]

                gen_dat_mask = get_generator_sample_mask(batch, gen_scores)
                gen_dis_scores = dis_scores[gen_dat_mask]
                sample_gen_scores = gen_scores[gen_dat_mask]

                rank_loss = compute_r_dis_loss(
                    true_dis_scores,
                    gen_dis_scores,
                    true_dat_mask,
                    sample_gen_scores,
                )

                ans_loss = ans_loss_fn(a_dis_scores, true_dat_mask.float())

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_rank_dis_loss += rank_loss.detach().float()
                    total_ans_dis_loss += ans_loss.detach().float()

                ans_loss = ans_loss / args.gradient_accumulation_steps
                rank_loss = rank_loss / args.gradient_accumulation_steps

                discriminator_engine.backward(rank_loss)
                discriminator_engine.step()
                r_dis_lr_scheduler.step()

                ans_discriminator_engine.backward(ans_loss)
                ans_discriminator_engine.step()
                a_dis_lr_scheduler.step()
                progress_bar.update(1)
                completed_steps += 1
                if completed_steps >= args.max_train_steps:
                    break

            ##########################
            # Evaluate Discriminator #
            ##########################
            eval_metrics = get_eval_metrics(
                args,
                generator_engine,
                discriminator_engine,
                ans_discriminator_engine,
                eval_dataloader,
            )
            if args.with_tracking:
                logger.log(
                    {
                        "epoch": epoch,
                        "d_epoch": d_epoch,
                        "train_rank_dis_loss": total_rank_dis_loss.item()
                        / len(train_dataloader),
                        "train_ans_dis_loss": total_ans_dis_loss.item()
                        / len(train_dataloader),
                        "step": completed_steps,
                        **eval_metrics,
                    },
                    step=completed_steps,
                )

            if rank == 0:
                torch.save(
                    {
                        "gen_model_state_dict": generator.state_dict(),
                        "dis_model_state_dict": discriminator.state_dict(),
                        "ans_model_state_dict": ans_discriminator.state_dict(),
                        "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                        "dis_optimizer_state_dict": dis_optimizer.state_dict(),
                        "ans_optimizer_state_dict": ans_optimizer.state_dict(),
                        "epoch": epoch,
                        "current_phase": "discriminator",
                        "starting_epoch": d_epoch + 1,
                    },
                    args.output_dir
                    + "epoch_{epoch}_generator_stage_{g_epoch}_loss_{eval_metrics[eval_loss]}",
                )
        # Set the starting discriminator epoch to 0 now that
        # we've completed all dis epochs
        if epoch == starting_epoch and skip_dis_phase:
            starting_dis_epoch = 0
        discriminator.eval()
        ans_discriminator.eval()
        generator.train()
        for g_epoch in range(starting_gen_epoch, args.num_gen_epochs):
            if args.with_tracking:
                total_gen_loss = 0
            for step, batch in enumerate(train_dataloader):
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        dis_scores = discriminator(**batch)["output_distribution"]
                        ans_scores = ans_discriminator(**batch)["output_distribution"]
                    gen_scores = generator(**batch)["output_idstribution"]
                    loss = compute_gen_loss(
                        batch,
                        gen_scores,
                        dis_scores,
                        ans_scores,
                        args.ans_discriminator_weight,
                        args.regularizer_weight,
                    )

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_gen_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    scaler.scale(loss).backward()
                    scaler.step(gen_optimizer)
                    gen_lr_scheduler.step()
                    gen_optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    scaler.update()
                if completed_steps >= args.max_train_steps:
                    break

            ##########################
            # Evaluate Generator #
            ##########################
            eval_metrics = get_eval_metrics(
                args,
                generator_engine,
                discriminator_engine,
                ans_discriminator_engine,
                eval_dataloader,
            )
            if args.with_tracking:
                logger.log(
                    {
                        "epoch": epoch,
                        "g_epoch": g_epoch,
                        "train_gen_loss": total_gen_loss.item() / len(train_dataloader),
                        "step": completed_steps,
                        **eval_metrics,
                    },
                    step=completed_steps,
                )

            if rank == 0:
                torch.save(
                    {
                        "gen_model_state_dict": generator.state_dict(),
                        "dis_model_state_dict": discriminator.state_dict(),
                        "ans_model_state_dict": ans_discriminator.state_dict(),
                        "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                        "dis_optimizer_state_dict": dis_optimizer.state_dict(),
                        "ans_optimizer_state_dict": ans_optimizer.state_dict(),
                        "epoch": epoch,
                        "current_phase": "generator",
                        "starting_epoch": g_epoch + 1,
                    },
                    args.output_dir
                    + "epoch_{epoch}_generator_stage_{g_epoch}_loss_{eval_metrics[eval_loss]}",
                )

    # Save final model state
    if rank == 0:
        perplexity = eval_metrics["perplexity"]
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity}, f)


def pepare_dataset(args, tokenizer):
    # Get the datasets: Loads a processed dataset from disk. Must be a dataset created by a DatasetProcessor
    raw_datasets = load_from_disk(args.dataset_path)
    # TODO remove once testing is over
    raw_datasets = DatasetDict(
        {
            "train": raw_datasets["train"].select(range(100)),
            "dev": raw_datasets["dev"].select(range(100)),
            "test": raw_datasets["test"].select(range(100)),
        }
    )

    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Answerability labeling for each document
    def find_whole_answer(w):
        return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search

    def label_answerability(example):
        compile_func = find_whole_answer(
            "|".join([re.escape(answer) for answer in example["answer"]])
        )
        answerability_labels = list(
            map(bool, (map(compile_func, example["documents"])))
        )
        return {"answerability_labels": answerability_labels}

    answerable_datasets = raw_datasets.map(
        label_answerability,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Labeling each document as having an answer or not",
    )

    # Keep only questions that have at least one answerable document
    answerable_datasets = answerable_datasets.filter(
        lambda example: any(example["answerability_labels"]),
        num_proc=args.preprocessing_num_workers,
    )

    # Tokenize the questions and documents
    def tokenize_function(examples):
        # Prepend each document with the question
        return tokenizer(
            [
                examples["question"] + tokenizer.sep_token + document
                for document in examples["documents"]
            ]
        )

    tokenized_datasets = answerable_datasets.map(
        tokenize_function,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["dev"]
    return train_dataset, eval_dataset


def get_eval_metrics(
    args, generator, discriminator, ans_discriminator, eval_dataloader
):
    len_dataset = len(eval_dataloader)
    generator.eval()
    discriminator.eval()
    ans_discriminator.eval()

    losses = []
    all_gen_scores = []
    all_dis_scores = []
    all_ans_scores = []
    all_answerability_labels = []
    for batch in eval_dataloader:
        with torch.cuda.amp.autocast():
            with torch.no_grad():

                gen_scores = generator(**batch)["output_distribution"]
                dis_scores = discriminator(**batch)["output_distribution"]
                ans_scores = ans_discriminator(**batch)["output_distribution"]
            loss = compute_gen_loss(
                batch,
                gen_scores,
                dis_scores,
                ans_scores,
                args.ans_discriminator_weight,
                args.regularizer_weight,
            )
        losses.append(loss.repeat(args.per_device_eval_batch_size))
        all_gen_scores.extend(gen_scores.tolist())
        all_dis_scores.extend(dis_scores.tolist())
        all_ans_scores.extend(ans_scores.tolist())
        all_answerability_labels.extend(batch["answerability_labels"].tolist())

    ret = {}
    ret.update(
        get_hits_dict(
            all_gen_scores, all_answerability_labels, args.hits_list, "generator"
        )
    )
    ret.update(
        get_hits_dict(
            all_dis_scores, all_answerability_labels, args.hits_list, "discriminator"
        )
    )
    ret.update(
        get_hits_dict(
            all_ans_scores,
            all_answerability_labels,
            args.hits_list,
            "ans_discriminator",
        )
    )

    losses = torch.cat(losses)
    losses = losses[:len_dataset]
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    ret["eval_loss"] = eval_loss
    ret["perplexity"] = perplexity
    return ret


def move_batch_to_devices(batch, gen_device, dis_device, ans_device):
    gen_batch = {
        key: value.to(gen_device) if key != "answerability_labels" else value
        for key, value in batch.items()
    }
    dis_batch = {
        key: value.to(dis_device) if key != "answerability_labels" else value
        for key, value in batch.items()
    }
    ans_batch = {
        key: value.to(ans_device) if key != "answerability_labels" else value
        for key, value in batch.items()
    }

    return gen_batch, dis_batch, ans_batch


if __name__ == "__main__":

    args = parse_args()

    deepspeed.init_distributed()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Ensure that the hits list is a list of integers
    if args.hits_list is not None:
        args.hit_list = [int(hit) for hit in args.hits_list]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset, eval_dataset = pepare_dataset(args, tokenizer)
    main(0, 1, train_dataset, eval_dataset)
