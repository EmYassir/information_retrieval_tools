import sys
import random
import numpy

from datasets import load_from_disk, concatenate_datasets
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="Creator of faiss index from shards")
    parser.add_argument("--n_shards", type=int, default=0,  help="The number of shards.") 
    parser.add_argument("--output_dir",  type=str, default="/media/data/yassir/output",   help="Output directory to save the faiss database." )
    parser.add_argument("--output_file_name", type=str, default="db")
    args = parser.parse_args()
    return args



def main():
     
    # Parser
    args = parse_args()
    if args.output_dir is None:
        raise ValueError("The output directory should be provided.")

    logger.info("Aggregating the shards ...")
    dataset_shards = []
    for rank in range(args.n_shards):
        dataset_shards.append(load_from_disk(args.output_dir + f"/wikipedia_split_encoded_shard_{rank}"))
    encoded_dataset = concatenate_datasets(dataset_shards)
    

    # Concatenating datasets
    logger.info("Saving the encoded dataset to disk ...")
    encoded_dataset.add_faiss_index("embeddings")
    encoded_dataset.save_faiss_index("embeddings", args.output_dir + "/" +  args.output_file_name + ".faiss")
    logger.info("Done")

    


if __name__ == '__main__':
    main()
