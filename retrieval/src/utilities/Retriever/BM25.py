import os
from .Retriever import Retriever
from datasets import Dataset
from rank_bm25 import BM25Okapi
import pickle
from typing import Optional


class BM25Retriever(Retriever):
    """
    Retriever module that constructs BM25 index and retrieves documents for QA Datasets.
    """

    BM25_CACHE_DIR = "/BM25Retriever_Files"

    def __init__(
        self,
        corpus: Dataset,
        output_dir: Optional[str] = None,
        index_path: Optional[str] = None,
        num_procs: Optional[int] = 32,
        ignore_cache_dirs: Optional[bool] = False,
    ):
        """Initializes teh BM25Retriever module

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
        if not output_dir and not index_path:
            raise ValueError("Output directory or index path must be specified")
        if output_dir.endswith("/"):
            output_dir = output_dir[:-1]
        self.corpus = corpus
        self.output_dir = output_dir
        self.index_path = index_path
        self.num_procs = num_procs
        self.ignore_cache_dirs = ignore_cache_dirs

        self.__is_initialized = False
        self.index = None

        self.init_cache_dirs(self.output_dir + self.BM25_CACHE_DIR)
        self.BM25_INDEX_REL_PATH = "/BM25Retriever_Files/BM25.pickle"
        if self.index_path or (
            not self.ignore_cache_dirs
            and os.path.exists(self.output_dir + self.BM25_INDEX_REL_PATH)
        ):
            self.__is_initialized = True

    @staticmethod
    def __get_top_k(example, corpus, index, k: int):
        return {
            "documents": index.get_top_n(example["encoded_query"], corpus["text"], k)
        }

    def get_docs(self, dataset: Dataset, k: int):
        """Retrieves the top k documents from the dataset containing encoded questions.

        Args:
            dataset (Dataset): Dataset with encoded questions
            k (int): Number of documents to get for each question.

        Returns:
            _type_: Dataset with k documents for each question.
        """
        return dataset.map(
            self.__get_top_k,
            fn_kwargs={"corpus": self.corpus, "index": self.index, "k": k},
            num_proc=self.num_procs,
            remove_columns="encoded_query",
        )

    @staticmethod
    def tokenize(example: dict) -> dict:
        """Used by map function to tokenize questions in the dataset into format used by BM25.

        Args:
            example (_type_): _description_

        Returns:
            _type_: _description_
        """
        return {"encoded_query": example["question"].lower().split(" ")}

    def prep_queries(self, dataset: Dataset) -> Dataset:
        """Adds column to dataset containing questions encoded in correct format for BM25 to perform Retrieval.

        Args:
            dataset (Dataset): QA Dataset. Assuming dataset contains question column

        Returns:
            Dataset: Original Dataset with new encoded_query column used for retrieval.
        """
        return dataset.map(self.tokenize, num_proc=self.num_procs)

    def build_index_for_corpus(self):
        """Constructs BM25 index if index has not been initialized yet. Overrides Retriever superclass function."""
        self.corpus = self.corpus.map(
            lambda example: {"tokenized_text": example["text"].lower().split(" ")}
        )
        self.index = BM25Okapi(self.corpus["tokenized_text"])
        with open(self.output_dir + self.BM25_INDEX_REL_PATH, "wb") as outfile:
            pickle.dump(self.index, outfile)

    def load_index_from_disk(self):
        """Loads index from path on disk

        Raises:
            RuntimeError: If no index is constructed or specified, throw an error
        """
        path = (
            self.index_path
            if self.index_path
            else self.output_dir + self.BM25_INDEX_REL_PATH
        )
        with open(path, "rb") as infile:
            self.index = pickle.load(infile)

    def is_initialized(self) -> bool:
        """Returns True if BM25 index has been loaded to gpus.

        Overrides superclass method.

        Returns:
            bool: True if index is loaded
        """
        return self.__is_initialized

    def __str__(self):
        return "bm25"
