import os
import sys
import socket
import random
import numpy
from tqdm import tqdm 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import pandas as pd

from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder
from transformers import DataCollatorWithPadding
import argparse
import logging
from threading import Lock

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def parse_args():
    parser = argparse.ArgumentParser(description="Parallel encoding of a dataset. Uses FB's DPR CTX Encoder (Huggingface implementation)")
    parser.add_argument("--n_gpu", type=int, default=0,  help="The number of available gpus.") 
    parser.add_argument("--tokenized_dataset_path", type=str, default=None,  help="The path of the tokenized dataset.") 
    parser.add_argument("--per_device_batch_size",  type=int, default=4,   help="Batch size (per device) for the dataloader." )
    parser.add_argument("--output_dir",  type=str, default="/media/data/yassir/output",   help="Output directory to save the faiss database." )
    parser.add_argument("--output_file_name", type=str, default="db")
    parser.add_argument("--seed",  type=int, default=33,   help="seed." )
    parser.add_argument("--dataloader_num_workers",  type=int, default=2)
    parser.add_argument("--dataloader_pin_memory",  type=bool, default=True)
    args = parser.parse_args()
    return args



def init_gpu_params(params):
    params.n_gpu = torch.cuda.device_count()
    if params.n_gpu <= 0:
        params.multi_gpu = False
        return

    print("Initializing GPUs")
    if params.n_gpu > 1:
        params.multi_gpu = True
    # local job (single GPU)
    else:
        params.multi_gpu = False

    


def train(args, local_rank, dataset, ctx_tokenizer, save_path='./'):
    processed_dataset = []
    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.n_gpu,
    	rank=local_rank
    )

    # Handle single and multi-GPU / multi-node.
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    if local_rank == 0:
        print("Loading Encoder...")
    
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_encoder.resize_token_embeddings(len(ctx_tokenizer))
    ctx_encoder.to(device)

    #print(f"===>>> Thread {local_rank}'s workload: {len(dataset)} samples to process.")

    # Synchronizing the processes
    dist.barrier()

    # Creating the dataloader
    if local_rank == 0:
        logger.info(f"Creating the dataloaders...")

    """
    sampler = DistributedSampler(
    	sub_dataset,
    	num_replicas=1,
        shuffle = False,
        rank = local_rank
    )
    """
    dataloader = DataLoader(dataset,
                batch_size=args.per_device_batch_size,
                shuffle=False,
                num_workers=args.dataloader_num_workers,
                pin_memory=args.dataloader_pin_memory,
                collate_fn=DataCollatorWithPadding(ctx_tokenizer),
                #sampler=sampler
            )
    if local_rank == 0:
        print(f"Dataloaders loaded...")
    # Synchronizing the processes
    #dist.barrier()

    # Encoding the data
    # Only show the progress bar once on each machine.
    iter_bar = tqdm(dataloader, desc="-Iter", disable=(local_rank != 0))
    for batch in iter_bar:
        with torch.no_grad():
            outputs = ctx_encoder(**{key: value.to(local_rank) for key,value in batch.items()}).pooler_output
        processed_dataset.extend(outputs.cpu().tolist())
        iter_bar.update()
    iter_bar.close()

    # Putting the result in the queue
    #queue.put((local_rank, processed_dataset))

    # Synchronizing the processes
    #dist.barrier()
    
    # Saving the dataset
    print(f"===>>> Saving the dataset shard {local_rank}...")
    df = pd.DataFrame({"embeddings": processed_dataset})
    dataset =  Dataset.from_pandas(df)
    dataset.save_to_disk(save_path)

    # Synchronizing the processes
    dist.barrier()
    
    # Synchronizing the threads and destroying them 
    dist.destroy_process_group()



def main():
     
    # Parser
    args = parse_args()
    if args.tokenized_dataset_path is None:
        raise ValueError("The dataset path should be provided.")



    # The tokenizer is gonna be useful for the dataloader
    logger.info("Loading Tokenizer...")
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    #ctx_tokenizer.share_memory()
    logger.info("Models loaded...")

    
    # ARGS and GPU initialization
    logger.info("Initializing parameters...")
    init_gpu_params(args)
    set_seed(args)


    logger.info("Encoding the datasets")
    processes = []
    ctx = mp.get_context('spawn')
    for rank in range(args.n_gpu):
        logger.info(f"=> Loading the dataset shard {rank}.")
        sub_dataset = load_from_disk(args.tokenized_dataset_path + f"_shard_{rank}")
        print(f"=> Thread {rank}'s workload: {len(sub_dataset)} samples to process.")
        p = ctx.Process(target=train, args=(args, rank, sub_dataset, ctx_tokenizer, args.output_dir + f"/wikipedia_split_encoded_with_title_shard_{rank}"))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    

    logger.info("Aggregating the shards ...")
    prefix = args.output_dir + "/wikipedia_split_encoded_with_title_shard_"
    dataset_shards = []
    for rank in range(args.n_gpu):
        dataset_shards.append(load_from_disk(prefix + str(rank)))
    encoded_dataset = concatenate_datasets(dataset_shards)
    print(f"Length of full dataset == {len(encoded_dataset)}")
    

    # Concatenating datasets
    logger.info("Saving the encoded dataset to disk ...")
    #encoded_dataset.add_faiss_index("embeddings", device=[0,1,2,3,4,5,6,7])
    encoded_dataset.add_faiss_index("embeddings")
    encoded_dataset.save_faiss_index("embeddings", args.output_dir + "/" +  args.output_file_name + ".faiss")
    logger.info("Done")

    


if __name__ == '__main__':
    main()
