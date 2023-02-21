#!/usr/bin/env python
import sys
import argparse
import logging
import os
import pprint
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


sys.path.insert(1, os.getcwd())
from src.model.utilities.trainer_utils import (
    compute_ans_loss,
    compute_r_dis_loss,
    compute_gen_loss,
    get_collate_fn,
    prepare_dataset,
    get_eval_metrics,
    move_batch_to_devices,
)
from src.model.gan_modeling import FullGANModel
from src.config.gan_config import FullGANConfig


logger = logging.getLogger(__name__)
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
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
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
        "--num_dis_rounds",
        type=int,
        default=3,
        help=("The number of discriminator epochs"),
    )
    parser.add_argument(
        "--num_gen_rounds",
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
        default=256,
        help=("Set the max length to truncate to."),
    )
    parser.add_argument(
        "--eval_on_step",
        type=int,
        default=10000,
        help=("Number of steps to print evaluation metrics"),
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
    generator = gan.generator.to(gen_device)
    discriminator = gan.discriminator.to(dis_device)
    ans_discriminator = gan.ans_discriminator.to(ans_device)

    # Data collator
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
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
    num_full_rounds = len(train_dataloader) // (
        args.num_dis_rounds + args.num_gen_rounds
    )
    max_dis_train_steps = args.num_train_epochs * num_full_rounds * args.num_dis_rounds
    max_gen_train_steps = args.num_train_epochs * num_full_rounds * args.num_gen_rounds

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

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )

    # Set the default starting epoch
    starting_epoch = 0

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

    scaler = torch.cuda.amp.GradScaler()
    dis_losses = []
    gen_losses = []
    for epoch in range(starting_epoch, args.num_train_epochs):
        generator.train()
        discriminator.train()
        ans_discriminator.train()

        round_counter = 0
        for step, batch in enumerate(tqdm(train_dataloader, disable=rank != 0)):
            dis_optimizer.zero_grad()
            ans_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            gen_batch, dis_batch, ans_batch = move_batch_to_devices(
                batch, gen_device, dis_device, ans_device
            )
            if round_counter < args.num_dis_rounds:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        gen_scores = generator(**gen_batch)["output_distribution"]
                    dis_scores = discriminator(**dis_batch)["output_distribution"]
                    ans_scores = ans_discriminator(**ans_batch)["output_distribution"]

                    rank_loss = compute_r_dis_loss(
                        batch, dis_scores, gen_scores, gen_device
                    )
                    ans_loss = compute_ans_loss(ans_scores, ans_batch, ans_device)

                scaler.scale(rank_loss).backward()
                scaler.scale(ans_loss).backward()

                # Update Discriminator
                scaler.step(dis_optimizer)
                r_dis_lr_scheduler.step()

                # Update Answerability Discriminator
                scaler.step(ans_optimizer)
                a_dis_lr_scheduler.step()

                scaler.update()
                round_counter += 1
                dis_losses.append(rank_loss.detach().item())
            else:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        dis_scores = discriminator(**dis_batch)["output_distribution"]
                        ans_scores = ans_discriminator(**ans_batch)[
                            "output_distribution"
                        ]
                    gen_scores = generator(**gen_batch)["output_distribution"]
                    loss = compute_gen_loss(
                        gen_batch,
                        gen_scores,
                        dis_scores,
                        ans_scores,
                        gen_device,
                        args.ans_discriminator_weight,
                        args.regularizer_weight,
                        # TODO: Add argument num_gen_samples
                        # args.num_gen_samples
                    )
                    gen_losses.append(loss.detach().item())
                scaler.scale(loss).backward()
                scaler.step(gen_optimizer)
                gen_lr_scheduler.step()
                scaler.update()

                if round_counter >= args.num_dis_rounds + args.num_gen_rounds - 1:
                    round_counter = 0
                else:
                    round_counter += 1
            if step == len(train_dataloader) - 1 or (
                step != 0 and step % args.eval_on_step == 0
            ):
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
                            **eval_metrics,
                        },
                        sort_dicts=False,
                    )
                )
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
                "gen_losses": gen_losses,
                "dis_losses": dis_losses,
            },
            args.output_dir + f"/epoch_{epoch}_loss_{round(eval_loss,2)}",
        )


if __name__ == "__main__":
    args = parse_args()

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Ensure that the hits list is a list of integers
    if args.hits_list is not None:
        args.hit_list = [int(hit) for hit in args.hits_list]

    # Get the dataset
    if args.is_tokenized:
        tokenized_datasets = load_from_disk(args.dataset_path)
    else:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        tokenized_datasets = prepare_dataset(
            args, BertTokenizerFast.from_pretrained("bert-base-uncased")
        )
    train_dataset, eval_dataset = tokenized_datasets["train"], tokenized_datasets["dev"]
    # Start training
    main(0, 1, train_dataset, eval_dataset)
