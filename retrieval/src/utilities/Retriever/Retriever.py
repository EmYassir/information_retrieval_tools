from abc import ABC, abstractmethod
from datasets import Dataset
from pathlib import Path


class Retriever(ABC):
    def init_cache_dirs(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_docs(self, dataset: Dataset, k: int):
        """
        Retrieves the top k documents for each question in the given dataset. The documents are
        extracted from teh Retriever's initialized corpus.

        Assumes the dataset contains the column: question
        """
        pass

    @abstractmethod
    def prep_queries(self, dataset: Dataset) -> Dataset:
        """
        Returns a dataset with a new column, encoded_query. This column contains the questions
        encoded in the correct format for the configured Retriever.

        Asumes the dataset contains the column: question
        """
        pass

    @abstractmethod
    def build_index_for_corpus(self):
        """
        Constructs and saves an index for the documents that exist in the corpus. The index is in
        the appropriate format for the configured Retriever.

        If there already exists a cache of the index or if one is already sepcified, a new one will
        not be built.
        """
        pass

    @abstractmethod
    def load_index_from_disk(self):
        """
        Loads a specified or cached index from the disk.
        """
        pass

    @abstractmethod
    def is_initialized(self):
        """Returns true if the index for document retrieval is initialized"""
        pass
