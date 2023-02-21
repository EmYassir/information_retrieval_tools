
import os
import sys
import argparse
from datasets import load_from_disk
import torch.multiprocessing as mp
import logging





def shard_func(args, shard, dataset):
    print(f"===>>> Processing shard {shard}...")
    sub_dataset = dataset['train'].shard(num_shards=args.n_shards, index=shard, contiguous=True)
    save_path = args.output_prefix + f"{shard}"
    print(f"===>>> Saving shard {shard} to disk ('{save_path}')...")
    sub_dataset.save_to_disk(save_path)



if __name__ == "__main__":
    # Logging
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # Parser
    parser = argparse.ArgumentParser(
        description="Dataset sharder"
    )
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--output_prefix", required=True, type=str)
    parser.add_argument("--n_shards", required=True, type=int)
    args = parser.parse_args()
    
    if args.n_shards < 1:
        raise ValueError('"n_shards" should be greater than 1.')

    if not os.path.isdir(args.dataset_path):
        raise ValueError('"dataset_path" should be a valid directory.')
    
    for shard in range(int(args.n_shards)):
        os.makedirs(args.output_prefix + f"{shard}", exist_ok = True)

    logger.info(f"## Loading the original dataet...")
    dataset = load_from_disk(args.dataset_path)

    print(">>>> [START] sharding the dataset ...")
    processes = []
    ctx = mp.get_context('spawn')
    for shard in range(0, args.n_shards):
        p = ctx.Process(target=shard_func, args=(args, shard, dataset))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    print("DONE")
