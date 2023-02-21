import sys
import os

sys.path.insert(1, os.getcwd())

import argparse
from datasets import load_from_disk, load_dataset
from src.utilities.DatasetProcessors.CuratedTRECProcessor import (
    CuratedTRECProcessor,
)
from src.utilities.DatasetProcessors.NaturalQuestionsProcessor import (
    NaturalQuestionsProcessor,
)
from src.utilities.DatasetProcessors.TriviaQAProcessor import TriviaQAProcessor
from src.utilities.DatasetProcessors.WebQuestionsProcessor import (
    WebQuestionsProcessor,
)
from src.utilities.Retriever.DPR import DPRRetriever
from src.utilities.Retriever.BM25 import BM25Retriever
from src.utilities.Retriever.Retriever import Retriever

NATURAL_QUESTIONS = "NaturalQuestions"
CURATED_TREC = "CuratedTREC"
TRIVIA_QA = "TriviaQA"
WEB_QUESTIONS = "WebQuestions"
DATASET_CHOICES = [NATURAL_QUESTIONS, CURATED_TREC, TRIVIA_QA, WEB_QUESTIONS]


def get_args():
    parser = argparse.ArgumentParser(
        description="Extracts the top  100 passages for each question. Returns answerable passages and Reader scores."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=DATASET_CHOICES,
        help="OpenQA Dataset to retrieve documents for.",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        help="Path to the chosen dataset. Must be the raw dataset path mentioned in the documentation of the DatasetProcessor to be used.",
    )
    parser.add_argument(
        "--retriever",
        required=True,
        type=str,
        choices=["dpr", "bm25"],
        help="Method of document retrieval",
    )
    parser.add_argument(
        "--corpus_path",
        required=True,
        type=str,
        help="Path to corpus to be used. Typically this is the DPR wikipedia corpus with the name psgs_w100.tsv",
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
        type=str,
        help="Directory used for Retriever caching and output datasets.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to output dataset with documents.",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        help="Path to saved index used by a Retriever. If index path not set, Retriever searches in the output_dir for the index. For DPR the following wikipedia corpus index is used database_bis.faiss.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=8,
        help="Number of GPUs to use for encoding or FAISS index. Not used in BM25 Retriever.",
    )
    parser.add_argument(
        "--k_passages",
        type=int,
        default=100,
        help="Number of passages to retrieve for each question",
    )
    parser.add_argument(
        "--num_procs",
        type=int,
        default=32,
        help="Number of processes to use for multiprocessing jobs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size to use for query encoding in GPUs. Not used in BM25 Retiever",
    )
    parser.add_argument(
        "--ignore_cache_dirs",
        action="store_true",
        help="Ignores the cache directories created by retriever.",
        default=False,
    )
    return parser.parse_args()


def get_retriever(corpus, args) -> Retriever:
    retriever = None
    if args.retriever == "dpr":
        retriever = DPRRetriever(
            corpus=corpus,
            output_dir=args.cache_dir,
            index_path=args.index_path,
            num_procs=args.num_procs,
            batch_size=args.batch_size,
            world_size=args.world_size,
            ignore_cache_dirs=args.ignore_cache_dirs,
        )
    elif args.retriever == "bm25":
        retriever = BM25Retriever(
            corpus=corpus,
            output_dir=args.cache_dir,
            index_path=args.index_path,
            num_procs=args.num_procs,
            ignore_cache_dirs=args.ignore_cache_dirs,
        )

    if not retriever or not retriever.is_initialized():
        raise RuntimeError(
            "Must pass output_dir or index_path that initializes the Retriever."
        )
    return retriever


def get_dataset_processor(args, retriever):
    if args.dataset == NATURAL_QUESTIONS:
        dataset_processor = NaturalQuestionsProcessor(
            args.dataset_path, retriever, args.k_passages
        )
    elif args.dataset == CURATED_TREC:
        dataset_processor = CuratedTRECProcessor(
            args.dataset_path, retriever, args.k_passages
        )
    elif args.dataset == TRIVIA_QA:
        dataset_processor = TriviaQAProcessor(
            args.dataset_path, retriever, args.k_passages
        )
    elif args.dataset == WEB_QUESTIONS:
        dataset_processor = WebQuestionsProcessor(
            args.dataset_path, retriever, args.k_passages
        )
    return dataset_processor


def get_corpus(args):
    if args.corpus_path.endswith(".tsv"):
        return load_dataset("csv", data_files=args.corpus_path, delimiter="\t")["train"]
    else:
        return load_from_disk(dataset_path=args.corpus_path)["train"]


if __name__ == "__main__":
    args = get_args()
    corpus = get_corpus(args)
    retriever = get_retriever(corpus, args)
    dataset_processor = get_dataset_processor(args, retriever)
    dataset_processor.prep_queries()
    dataset_processor.get_documents()
    dataset_processor.save_document_dataset(args.output_dir)
