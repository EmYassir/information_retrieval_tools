import os
import torch
import torch.distributed as dist
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder
from functools import reduce

from transformers import DataCollatorWithPadding

WIKI_CORPUS_PATH = "/media/data/yassir/datasets/downloads/data/wikipedia_split/psgs_w100.tsv"
FAISS_INDEX_PATH ="/media/data/yassir/datasets/faiss/wiki_corpus.faiss"
K_VALUE=100
WORLD_SIZE = 2



def cleanup():
    dist.destroy_process_group()
def get_passage_mask(example, batch_passages):
    return [[reduce(lambda a,b: a or b in passage,example[idx]["answer"])]for nearest_example, idx in enumerate(batch_passages.total_examples) for passage in nearest_example["text"]]
def get_masked_answerable_passages(batch_passages, mask):
    answerable_batch_dict = {"total_scores": [], "total_examples": []}
    for i, nearest_example in enumerate(batch_passages.total_scores):
        print(i)
        scores = []
        total_examples = {"id": [], "text": [], "title": []}
        for j, passage in enumerate(nearest_example):
            if mask[i][j]:
                scores.append(batch_passages.total_scores[i][j])
                total_examples["id"].append(batch_passages.total_examples[i]["id"][j])
                total_examples["text"].append(batch_passages.total_examples[i]["text"][j])
                total_examples["title"].append(batch_passages.total_examples[i]["title"][j])
        answerable_batch_dict["total_scores"].append(np.array(scores))
        answerable_batch_dict["total_examples"].append(total_examples)
    return answerable_batch_dict
def get_answerable_passages_from_top_k(example, corpus, rank, q_encoder, q_tokenizer):
    with torch.no_grad():
        encodings =  q_encoder(**{key: value.to(rank)
                                  for key, value in q_tokenizer(
                                        example["question"]
                                        ,padding=True,
                                        truncation=True,
                                        return_tensors="pt").items()
                                  }).pooler_output
    batch_passages = corpus.get_nearest_examples_batch("embedding", encodings.detach().numpy(),K_VALUE)
    passage_mask = get_passage_mask(example, batch_passages)
    return get_masked_answerable_passages(batch_passages,passage_mask)
def main(rank, world_size, dataset, corpus, queue):
    # setup the process groups
    setup(rank, world_size)
    print("Initializing model")

    #Setup Encoder
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(rank)

    # wrap the model with DDP
    q_encoder = DDP(q_encoder, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    processed_shard = dataset.shard(
                        num_shards=world_size,
                        index=rank,contiguous=True
                    ).map(
                        get_answerable_passages_from_top_k,
                        fn_kwargs={"corpus": corpus, "rank": rank, "q_encoder": q_encoder, "q_tokenizer": q_tokenizer},
                        batched=True,
                        batch_size=256
                    )
    queue.put((rank, processed_shard))
    cleanup()


def load_val_dataset():
    nq_open = load_dataset("nq_open")
    val = nq_open["validation"]
    return val
def map_dataset_encoding(dataset, corpus):

    ctx = mp.get_context('spawn')
    queue = ctx.SimpleQueue()
    for rank in range(WORLD_SIZE):
        ctx.Process(target=main, args=(rank, WORLD_SIZE, dataset, corpus, queue)).start()
    dataset_shards = [None]*WORLD_SIZE
    for rank in range(WORLD_SIZE):
        temp_result = queue.get()
        dataset_shards[temp_result[0]] = temp_result[1]
        del temp_result
    dataset = concatenate_datasets(dataset_shards)
    return dataset

if __name__ == '__main__':
    # Corpus
    corpus = load_dataset("csv", data_files=WIKI_CORPUS_PATH, delimiter="\t")
    corpus = corpus["train"]
    corpus.load_faiss_index("embedding",FAISS_INDEX_PATH)

    # Val datset
    nq_open = load_dataset("nq_open")
    val_dataset = nq_open["validation"]
    val_dataset = val_dataset.select(range(516))
    output_dataset = map_dataset_encoding(val_dataset, corpus)
    print("OUTPUT DATASET:")
    print("==================================")
    print(output_dataset)
    output_dataset.save_to_disk("answerable_passages_dpr")

