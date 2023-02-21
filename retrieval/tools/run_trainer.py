#!/usr/bin/env python
from re import T
import sys
import argparse
import logging
import math
import os
import pprint
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
from datasets import load_from_disk

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    MODEL_MAPPING,
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
    BertTokenizerFast,
)
from transformers.utils.versions import require_version

sys.path.insert(1, os.getcwd())
from src.model.utilities.trainer_utils import (
    compute_gen_ind_loss,
    compute_r_dis_loss,
    compute_gen_loss,
    get_collate_fn,
    get_eval_metrics,
    prepare_dataset,
    compute_ans_loss,
    move_batch_to_devices,
)
from src.model.gan_modeling import FullGANModel
from src.config.gan_config import FullGANConfig
from transformers import AutoModelForSequenceClassification

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
        "--is_tokenized",
        action="store_true",
        help="If set to true, the dataset is assumed to be tokenized and tokenization is not performed",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=True,
        help="Path to the training configuration with gernator, rank_discriminator, and answerability_discriminator configurations.",
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
        default=1e-5,
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
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated. Only used if is_tokenized not set"
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
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
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=350,
        help=("Set the max length to truncate to."),
    )
    parser.add_argument(
        "--eval_on_step",
        type=int,
        default=10000,
        help=("Number of steps to print evaluation metrics"),
    )
    parser.add_argument(
        "--checkpoint_model",
        action="store_true",
        help=("Set to true to ignore checkpointing"),
    )
    parser.add_argument(
        "--tensor_board", type=str, help=("Path to the tensorboard run")
    )
    args = parser.parse_args()

    return args


def main(rank, world_size, train_dataset, eval_dataset):
    args = parse_args()
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

    # Load and initialize the models
    gan_config = FullGANConfig.from_json_file(args.config_path)
    gan = FullGANModel(gan_config)
    gen_device = torch.device("cuda:0")
    dis_device = torch.device("cuda:1")
    ans_device = torch.device("cuda:2")

    model_name = "bert-base-uncased"
    generator = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(gen_device)
    discriminator = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(dis_device)
    ans_discriminator = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(
        ans_device
    )  # E.g. model was saved using `save_pretrained('./test/saved_model/')`

    # Data collator
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
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
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_dis_train_steps = (
        args.num_train_epochs * num_update_steps_per_epoch * args.num_dis_epochs
    )
    max_gen_train_steps = (
        args.num_train_epochs * num_update_steps_per_epoch * args.num_gen_epochs
    )
    max_train_steps = max_dis_train_steps + max_gen_train_steps

    def init_optimizer(model, train_steps):
        """Initializes optimizer and learning rate scheduler

        Args:
            model (torch.nn.module): Module to initialize optimizer and lr_scheduler for
            train_steps (int): Number of train steps expected. Used to initialize the scheduler

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
            num_training_steps=train_steps,
        )
        return optimizer, lr_scheduler

    # Initialize the optimizer for generator, discriminator, and answer discriminator
    gen_optimizer, gen_lr_scheduler = init_optimizer(generator, max_gen_train_steps)
    dis_optimizer, r_dis_lr_scheduler = init_optimizer(
        discriminator, max_dis_train_steps
    )
    ans_optimizer, a_dis_lr_scheduler = init_optimizer(
        ans_discriminator, max_dis_train_steps
    )

    # Initialize scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

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
    logger.info(f"  Total optimization steps = {max_train_steps}")

    starting_epoch = 0
    starting_dis_epoch = 0
    starting_gen_epoch = 0
    skip_dis_phase = False
    if args.tensor_board is not None:
        writer = SummaryWriter(args.tensor_board)
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            logger.info(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
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

    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.resume_from_checkpoint and not epoch == starting_epoch:
            starting_dis_epoch = 0
            starting_gen_epoch = 0
        #######################
        # Discriminator Stage #
        #######################
        generator.eval()
        discriminator.train()
        ans_discriminator.train()
        for d_epoch in range(starting_dis_epoch, args.num_dis_epochs):
            total_rank_dis_loss = 0
            total_ans_dis_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {key: value[0] for key, value in batch.items()}
                ans_optimizer.zero_grad()
                dis_optimizer.zero_grad()
                gen_batch, dis_batch, ans_batch = move_batch_to_devices(
                    batch, gen_device, dis_device, ans_device
                )
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        gen_scores = generator(
                            input_ids=gen_batch["input_ids"],
                            attention_mask=gen_batch["attention_mask"],
                            token_type_ids=gen_batch["token_type_ids"],
                        ).logits.squeeze(-1)
                    dis_scores = discriminator(
                        input_ids=dis_batch["input_ids"],
                        attention_mask=dis_batch["attention_mask"],
                        token_type_ids=dis_batch["token_type_ids"],
                    ).logits.squeeze(-1)
                    ans_scores = ans_discriminator(
                        input_ids=ans_batch["input_ids"],
                        attention_mask=ans_batch["attention_mask"],
                        token_type_ids=ans_batch["token_type_ids"],
                    ).logits.squeeze(-1)
                    rank_loss = compute_r_dis_loss(
                        batch, dis_scores, gen_scores, gen_device
                    )
                    ans_loss = compute_ans_loss(ans_scores, ans_batch, ans_device)

                    # We keep track of the loss at each epoch
                    total_rank_dis_loss += rank_loss.detach().float()
                    total_ans_dis_loss += ans_loss.detach().float()
                if args.tensor_board is not None:
                    writer.add_scalar(
                        "Dsicriminator loss",
                        rank_loss.detach().item(),
                        (epoch + d_epoch) * len(train_dataloader) + step,
                    )
                    writer.add_scalar(
                        "Answer_Discriminator loss",
                        ans_loss.detach().item(),
                        (epoch + d_epoch) * len(train_dataloader) + step,
                    )

                scaler.scale(rank_loss).backward()
                scaler.scale(ans_loss).backward()
                # Updates the scale for next iteration.
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    scaler.step(dis_optimizer)
                    r_dis_lr_scheduler.step()
                    scaler.step(ans_optimizer)
                    a_dis_lr_scheduler.step()
                    scaler.update()

            ##########################
            # Evaluate Discriminator #
            ##########################
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
            logger.info(
                msg=pprint.pformat(
                    {
                        "epoch": epoch,
                        "d_epoch": d_epoch,
                        **eval_metrics,
                        "mean_discriminator_loss": total_rank_dis_loss.item()
                        / len(train_dataloader),
                        "mean_ans_discriminator_loss": total_ans_dis_loss.item()
                        / len(train_dataloader),
                    },
                    sort_dicts=False,
                )
            )

            if rank == 0 and args.checkpoint_model:
                eval_loss = eval_metrics["eval_loss"]
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
                    + f"/epoch_{epoch}_discriminator_stage_{d_epoch}_loss_{round(eval_loss,2)}",
                )
        # Set the starting discriminator epoch to 0 now that
        # we've completed all dis epochs after resuming
        if epoch == starting_epoch and skip_dis_phase:
            starting_dis_epoch = 0
        ###################
        # Generator stage #
        ###################
        discriminator.eval()
        ans_discriminator.eval()
        generator.train()
        for g_epoch in range(starting_gen_epoch, args.num_gen_epochs):
            total_gen_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                gen_optimizer.zero_grad()
                batch = {key: value[0] for key, value in batch.items()}
                gen_batch, dis_batch, ans_batch = move_batch_to_devices(
                    batch, gen_device, dis_device, ans_device
                )
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        dis_scores = discriminator(
                            input_ids=dis_batch["input_ids"],
                            attention_mask=dis_batch["attention_mask"],
                            token_type_ids=dis_batch["token_type_ids"],
                        ).logits.squeeze(-1)
                        ans_scores = ans_discriminator(
                            input_ids=ans_batch["input_ids"],
                            attention_mask=ans_batch["attention_mask"],
                            token_type_ids=ans_batch["token_type_ids"],
                        ).logits.squeeze(-1)
                    gen_scores = generator(
                        input_ids=gen_batch["input_ids"],
                        attention_mask=gen_batch["attention_mask"],
                        token_type_ids=gen_batch["token_type_ids"],
                    ).logits.squeeze(-1)
                    loss = compute_gen_loss(
                        gen_batch,
                        gen_scores,
                        dis_scores,
                        ans_scores,
                        gen_device,
                        args.ans_discriminator_weight,
                        args.regularizer_weight,
                    )
                    # We keep track of the loss at each epoch
                    total_gen_loss += loss.detach().float()
                if args.tensor_board is not None:
                    writer.add_scalar(
                        "Generator loss",
                        loss.detach().item(),
                        (epoch + g_epoch) * len(train_dataloader) + step,
                    )
                scaler.scale(loss).backward()
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    scaler.step(gen_optimizer)
                    gen_lr_scheduler.step()
                    scaler.update()

            ##########################
            # Evaluate Discriminator #
            ##########################
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
            logger.info(
                msg=pprint.pformat(
                    {
                        "epoch": epoch,
                        "g_epoch": g_epoch,
                        **eval_metrics,
                        "gen_loss": total_gen_loss.item() / len(train_dataloader),
                    },
                    sort_dicts=False,
                )
            )

            if rank == 0 and args.checkpoint_model:
                eval_loss = eval_metrics["eval_loss"]
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
                    + f"/epoch_{epoch}_generator_stage_{g_epoch}_loss_{round(eval_loss,2)}",
                )
        # Set the starting generator epoch to 0 now that
        # we've completed all gen epochs after resuming
        if epoch == starting_epoch and not skip_dis_phase:
            starting_gen_epoch = 0


if __name__ == "__main__":
    args = parse_args()

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Ensure that the hits list is a list of integers
    if args.hits_list is not None:
        args.hit_list = [int(hit) for hit in args.hits_list]

    if args.is_tokenized:
        tokenized_datasets = load_from_disk(args.dataset_path)
    else:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        tokenized_datasets = prepare_dataset(
            args, BertTokenizerFast.from_pretrained("bert-base-uncased")
        )
    train_dataset, eval_dataset = tokenized_datasets["train"], tokenized_datasets["dev"]
    main(0, 1, train_dataset, eval_dataset)
