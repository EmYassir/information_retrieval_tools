import os
import sys
from .Retriever import Retriever
import torch
import torch.distributed as dist
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    DPRQuestionEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRContextEncoderTokenizerFast,
    DPRContextEncoder,
)
from src.utilities.Retriever.retriever_utils import (
    multi_gpu_perform,
    setup,
    prepare,
    cleanup,
)
import logging
from faiss import (
    GpuMultipleClonerOptions,
    index_cpu_to_gpus_list,
    index_cpu_to_all_gpus,
    read_index,
    Index,
)
from datasets import concatenate_datasets
import numpy as np
from typing import Optional, Union, List

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


class DPRRetriever(Retriever):
    """
    Retriever module that constructs Faiss index and retrieves documents for QA Datasets using DPR.
    """

    DPR_CACHE_DIR = "/DPR_Retriever_files"

    def __init__(
        self,
        corpus: Dataset,
        output_dir: str,
        index_path: Optional[str] = None,
        num_procs: Optional[int] = 32,
        batch_size: Optional[int] = 512,
        world_size: Optional[int] = 8,
        faiss_gpu_list: Optional[List[int]] = None,
        ignore_cache_dirs: Optional[bool] = False,
    ):
        """Initializes the DPRRetriever module

        Args:
            corpus (Dataset): Corpus containing passages. Must contain text column
            output_dir (Optional[str], optional): Directory to output to. Must specify either output_dir or index_path. Defaults to None.
            index_path (Optional[str], optional): Path to the Faiss index for the corpus. Must specify either output_dir or index_path. Defaults to None.
            num_procs (Optional[int], optional): Number of processed to use. Defaults to 32.
            batch_size (Optional[int], optional): Batch size to use for document and query enocding. Defaults to 512.
            world_size (Optional[int], optional): Number of GPUs to use for Faiss index and encoding. Defaults to 8.
            ignore_cache_dirs (Optional[bool], optional): Set to True to ignore existing cache in the output directory. Defaults to False.

        Raises:
            ValueError: Raised if neither output_dir or index_path is set.
        """
        super().__init__()
        if output_dir.endswith("/"):
            output_dir = output_dir[:-1]

        self.corpus = corpus

        self.__is_initialized = False
        self.__is_tokenized = False
        self.__index = None

        self.__output_dir = output_dir
        self.__num_procs = num_procs
        self.__batch_size = batch_size
        self.__world_size = world_size
        self.__ignore_cache_dirs = ignore_cache_dirs
        self.__index_path = index_path
        self.__faiss_gpu_list = faiss_gpu_list
        self.__FAISS_REL_PATH = "/DPR_Retriever_files/DPR_index.faiss"

        self.__q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.__ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )

        # Ensure that models are downloaded before processing
        DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

        self.init_cache_dirs(self.__output_dir + self.DPR_CACHE_DIR)

        if self.__index_path or (
            not self.__ignore_cache_dirs
            and os.path.exists(self.__output_dir + self.__FAISS_REL_PATH)
        ):
            self.__is_initialized = True

    def build_index_for_corpus(self):
        """
        Constructs Faiss index if index has not been initialized yet. Overrides Retriever superclass function.
        """
        if self.__is_initialized:
            return
        if not self.__is_tokenized:
            self.tokenize_corpus()
        encoding_ds = self.__encode_corpus("corpus")
        self.__construct_index(encoding_ds)
        self.__is_initialized = True
        self.retire_index()

    def __construct_index(self, encoding_ds: Dataset):
        """Constructs and saves faiss index from dataset.

        Args:
            encoding_ds (Dataset): Dataset containing embeddings column.
        """
        encoding_ds.add_faiss_index(column="embeddings")
        encoding_ds.save_faiss_index(
            "embeddings", self.__output_dir + self.__FAISS_REL_PATH
        )

    def tokenize_corpus(self):
        """Tokenizes the documents in the corpus"""

        def tokenize_func(example):
            return self.__ctx_tokenizer(
                example["title"], example["text"], truncation=True
            )

        self.corpus = self.corpus.map(tokenize_func, num_proc=self.__num_procs)
        self.corpus.set_format(
            columns=["input_ids", "attention_mask", "token_type_ids"], type="torch"
        )
        self.__is_tokenized = True

    def __encode_corpus(self, dataset_name: Optional[str] = "") -> Dataset:
        """Uses multigpu to encode the documents in a corpus

        Returns:
            Dataset: Returns dataset with encoding column
        """
        return multi_gpu_perform(
            self.__world_size,
            self.corpus,
            self.encode_documents,
            self.__output_dir + self.DPR_CACHE_DIR,
            self.__batch_size,
            dataset_name,
            self.__ignore_cache_dirs,
        )

    @staticmethod
    def encode_documents(
        rank: int, world_size: int, dataset: Dataset, prefix: str, batch_size: int
    ):
        """GPU function to encode documents in the corpus using the DPR context encoder.


        Args:
            rank (int): Rank of the process running function
            world_size (int): Total number of processes
            dataset (Dataset): Dataset to perform function on. Assumes the following columns exist: input_ids, attention_mask
            prefix (str): Prefix of path to cache the function outputs
            batch_size (int): Batch size to use for data
        """
        # setup the process groups
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Setup Encoder
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        ).to(device)

        dataset = dataset.shard(num_shards=world_size, index=rank, contiguous=True)
        dataloader = prepare(dataset, tokenizer, batch_size=batch_size)
        encoding_col = []

        if rank == 0:
            logger.info(f"Dataloaders loaded...")

        # Synchronizing the processes
        dist.barrier()

        # Encoding the data
        # Only show the progress bar once on each machine.
        iter_bar = tqdm(dataloader, desc="-Iter", disable=rank != 0)
        for batch in iter_bar:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = encoder(**batch).pooler_output
            encoding_col.extend(outputs.tolist())
            iter_bar.update()
        iter_bar.close()

        df = pd.DataFrame({"embeddings": encoding_col})
        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk(prefix + str(rank))

        dist.barrier()
        cleanup()

    @staticmethod
    def encode_query(
        rank: int, world_size: int, dataset: Dataset, prefix: str, batch_size: int
    ):
        """GPU function to encode the question for a QA Dataset

        Args:
            rank (int): Rank of the process running function
            world_size (int): Total number of processes
            dataset (Dataset): Dataset to perform function on. Assumes the following columns exist: input_ids, attention_mask
            prefix (str): Prefix of path to cache the function outputs
            batch_size (int): Batch size to use for data
        """

        # Setup the process groups
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Setup Encoder
        tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        ).to(device)

        dataset = dataset.shard(num_shards=world_size, index=rank, contiguous=True)
        dataloader = prepare(dataset, tokenizer, batch_size=batch_size)
        encoding_col = []

        if rank == 0:
            logger.info(f"Dataloaders loaded...")

        # Synchronizing the processes
        dist.barrier()

        # Encoding the data
        # Only show the progress bar once on each machine.
        iter_bar = tqdm(dataloader, desc="-Iter", disable=rank != 0)
        for batch in iter_bar:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = encoder(**batch).pooler_output
            encoding_col.extend(outputs.tolist())
            iter_bar.update()
        iter_bar.close()

        df = pd.DataFrame({"encoded_query": encoding_col})
        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk(prefix + str(rank))

        dist.barrier()
        cleanup()

    def __get_faiss_index(
        self, path: str, gpu_list: Optional[Union[int, List]] = None
    ) -> Index:
        """Reads index from path and loads to gpus

        Args:
            path (str): Path containing the Faiss index
            gpu_list (Optional[Union[int, List]], optional): List of GPU ids to load Faiss index to. If set to -1, use all
            available GPUs. Defaults to None.

        Returns:
            Index: Faiss index that is loaded to GPUs
        """
        index = read_index(path)
        co = GpuMultipleClonerOptions()
        co.shard = True
        if gpu_list:
            return index_cpu_to_gpus_list(index=index, co=co, gpus=gpu_list)
        else:
            return index_cpu_to_all_gpus(index=index, co=co)

    @staticmethod
    def __get_top_k(example: dict, index: Index, k: int) -> dict:
        """Function to used by map to retrieve top k documents from a loaded index

        Args:
            example (dict): Datum from the dataset being processed
            index (Index): Index to perform retrieval on
            k (int): Number of documents to retrieve

        Returns:
            dict: New column of indices of the top k documents in the corpus
        """
        _scores, indices = index.search(np.array(example["encoded_query"]), k)
        return {"indices": indices}

    @staticmethod
    def __get_passage_from_indices(example: dict, corpus: Dataset) -> dict:
        """Function used by map to get indices of top k documents from the corpus

        Args:
            example (dict): Datum from the dataset being processed
            corpus (Dataset): Corpus containing documents

        Returns:
            dict: Documents from the corpus at the top k indices
        """
        rows = corpus.select(example["indices"])
        documents = rows["text"]
        titles = rows["title"]
        return {"documents": documents, "titles": titles}

    def get_docs(self, dataset: Dataset, k: int) -> Dataset:
        """Retrieves the top k documents from the dataset containing encoded questions.

        Args:
            dataset (Dataset): Dataset with encoded questions
            k (int): Number of documents to get for each question.

        Returns:
            _type_: Dataset with k documents for each question.
        """
        if not self.__is_initialized:
            raise RuntimeError(
                "Retriever Uninitialized: Cannot load from disk when Retriever is not initialized"
            )
        if not self.__index:
            self.load_index_from_disk()
        dataset.set_format(
            columns="encoded_query",
            type="numpy",
            format_kwargs={"dtype": np.float32},
            output_all_columns=True,
        )
        dataset_w_indices = dataset.map(
            self.__get_top_k,
            fn_kwargs={"index": self.__index, "k": k},
            batched=True,
            remove_columns="encoded_query",
        )
        self.retire_index()
        dataset_w_passages = dataset_w_indices.map(
            self.__get_passage_from_indices,
            fn_kwargs={"corpus": self.corpus},
            num_proc=self.__num_procs,
            #remove_columns="indices",
        )
        dataset_w_passages.set_format(type=None)
        return dataset_w_passages

    @staticmethod
    def __tokenize_func(
        example: dict, q_tokenizer: DPRQuestionEncoderTokenizerFast
    ) -> dict:
        """Function used by map to tokenize the questions in the dataset

        Args:
            example (dict): Datum from the dataset being processed
            q_tokenizer (DPRQuestionEncoderTokenizerFast): DPR question tokenizer

        Returns:
            dict: Out tokenization of questions. Contains input_ids, token_type_ids, and attention_mask
        """
        return q_tokenizer(example["question"], truncation=True)

    def __tokenize_questions(self, dataset: Dataset):
        """Tokenizes the question column in the dataset and adds the columns input_ids,
        attention_mask, and token_type_ids

        Args:
            dataset (Dataset): Dataset with a quesiton column

        Returns:
            _type_: Dataset with new columns: input_ids, attention_mask, and token_type_ids
        """
        dataset = dataset.map(
            self.__tokenize_func,
            fn_kwargs={"q_tokenizer": self.__q_tokenizer},
            num_proc=self.__num_procs,
        )
        dataset.set_format(
            columns=["input_ids", "attention_mask", "token_type_ids"], type="torch"
        )
        return dataset

    def __encode_questions(
        self, dataset: Dataset, dataset_name: Optional[str] = ""
    ) -> Dataset:
        """Uses multiple GPUs to encode the questions in the dataset

        Args:
            dataset (Dataset): QA Dataset to encode the questions for. Assumes there exists a question column.
            dataset_name (Optional[str]): Name of dataset for caching
        Returns:
            Dataset: Dataset containing a column of the embeddings of the questions
        """
        embeddings = multi_gpu_perform(
            self.__world_size,
            dataset,
            self.encode_query,
            self.__output_dir + self.DPR_CACHE_DIR,
            self.__batch_size,
            dataset_name,
            self.__ignore_cache_dirs,
        )
        if len(dataset) != len(embeddings):
            raise RuntimeError(
                f"Dataset and Embedding lengths do not match. Dataset has {len(dataset)} elements. Embeddings has {len(embeddings)} elements"
            )
        print("Dataset Lengeth: ", len(dataset))
        print("Embeddings length: ", len(embeddings))
        return concatenate_datasets([dataset, embeddings], axis=1)

    def prep_queries(self, dataset: Dataset, dataset_name: Optional[str]) -> Dataset:
        """Adds column to dataset containing questions encoded in correct format for DPR to perform Retrieval.

        Args:
            dataset (Dataset): QA Dataset. Assuming dataset contains question column
            dataset_name (str): Name of dataset for caching
        Returns:
            Dataset: Original Dataset with new encoded_query column used for retrieval.
        """
        dataset = self.__tokenize_questions(dataset)
        dataset = self.__encode_questions(dataset, dataset_name)
        return dataset.remove_columns(["input_ids", "attention_mask", "token_type_ids"])

    def load_index_from_disk(self):
        """Loads index from path on disk

        Raises:
            RuntimeError: If no index is constructed or specified, throw an error
        """
        if self.__index:
            return
        elif self.__is_initialized:
            path = (
                self.__index_path
                if self.__index_path
                else self.__output_dir + self.__FAISS_REL_PATH
            )
            self.__index = self.__get_faiss_index(path, self.__faiss_gpu_list)
        else:
            raise RuntimeError(
                "Retriever Uninitialized: Cannot load from disk when Retriever is not initialized"
            )
        # TODO: Verify index and corpus size match. Raise error if not the case

    def retire_index(self):
        """Removes the Faiss index to free resources"""
        self.__index = None

    def is_initialized(self) -> bool:
        """Returns True if there is a FAISS index on disk.

        Overrides superclass method.

        Returns:
            bool: True if index is loaded
        """
        return self.__is_initialized

    def __str__(self):
        return "dpr"
