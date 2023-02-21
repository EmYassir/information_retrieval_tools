import os
import shutil
from datasets import load_from_disk, concatenate_datasets
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from typing import Callable, Optional


def multi_gpu_perform(
    world_size: int,
    dataset: Dataset,
    perform_func: Callable,
    output_dir: str,
    batch_size: int,
    dataset_name: str,
    ignore_cached_dir: Optional[bool] = False,
):
    """Performs the perform_func over multiple GPUs and aggregates the results using sharding. Assumes the perform func
    appropriately sets up and cleans process group, and correctly saves dataset shards in correct directory and format.

    Args:
        world_size (int): Total number of GPUs
        dataset (Dataset): Dataset to shard and operate on
        perform_func (Callable): Function to perform on dataset shards
        output_dir (str): Directory to output to
        batch_size (int): Batch size of GPU operation
        ignore_cached_dir (Optional[bool], optional): Set to true if cached shards should be ignored. Defaults to False.

    Returns:
        _type_: _description_
    """
    prefix = (
        output_dir
        + "/dataset_shard_"
        + perform_func.__name__
        + "_"
        + dataset_name
        + "_"
    )
    processed_shards_exists = True
    for rank in range(world_size):
        if not os.path.exists(prefix + str(rank)):
            processed_shards_exists = False
            break
    if ignore_cached_dir or not processed_shards_exists:
        mp.spawn(
            perform_func,
            args=(world_size, dataset, prefix, batch_size),
            nprocs=world_size,
            join=True,
        )
    dataset_shards = []
    for rank in range(world_size):
        dataset_shards.append(load_from_disk(prefix + str(rank)))
    dataset = concatenate_datasets(dataset_shards).flatten_indices()
    return dataset


def clean_shards(world_size: int, output_dir: str, perform_func: Callable):
    """Cleans cached shards created by multi gpu perform

    Args:
        world_size (int): Total number of shards created
        output_dir (str): Cache directory
        perform_func (Callable): Function to perform on the shards
    """
    prefix = output_dir + "/dataset_shard_" + perform_func.__name__ + "_"
    for rank in range(world_size):
        if os.path.exists(prefix + str(rank)):
            shutil.rmtree(prefix + str(rank))


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: Optional[int] = 32,
    pin_memory: Optional[bool] = False,
    num_workers: Optional[int] = 2,
):
    """Creates dataloader for multi gpu perform

    Args:
        dataset (Dataset): Dataset to create Dataloader for
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to tokenize the dataset
        batch_size (Optional[int], optional): Batch size to use. Defaults to 32.
        pin_memory (Optional[bool], optional): Defaults to False.
        num_workers (Optional[int], optional): Number of workers. Defaults to 2.

    Returns:
        _type_: _description_
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer, padding=True),
    )
    return dataloader
