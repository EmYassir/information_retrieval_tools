from abc import ABC, abstractmethod, abstractproperty
from datasets import DatasetDict
from src.utilities.Retriever.Retriever import Retriever


class OpenQADatasetProcessor(ABC):
    """
    Superclass for OpenQA Datasets that contain questions and answers. Deals with
    document retrieval by taking initialized retriever as input"""

    def __init__(self, retriever: Retriever, k: int = 100):
        """Initializes and loads QA dataset that does not contain documents. Uses an initialized
        Retriever module to retrieve documents for each question in the dataset.


        Args:
            retriever (Retriever): Initialized Retriever module.
            k (int, optional): Number of passages to retreive. Defaults to 100.

        Raises:
            ValueError: Throws error if the Retriever is not initialized
        """
        if not retriever.is_initialized():
            raise ValueError("Must provide an initialized retriever")
        self.__contains_documents = False
        self.retriever = retriever
        self.k = k
        self.ds = None
        self.encoded_ds = None

    def get(self) -> DatasetDict:
        if self.__contains_documents:
            return self.ds
        else:
            raise RuntimeError(
                "Docs Not Processed: Must retrieve documents before saving"
            )

    def prep_queries(self):
        train = self.retriever.prep_queries(
            self.ds_no_docs["train"], self.dataset_name + "_train"
        )
        dev = self.retriever.prep_queries(
            self.ds_no_docs["dev"], self.dataset_name + "_dev"
        )
        test = self.retriever.prep_queries(
            self.ds_no_docs["test"], self.dataset_name + "_test"
        )
        self.encoded_ds = DatasetDict({"train": train, "dev": dev, "test": test})

    def get_documents(self):
        """Uses Retriever to get documents for each question in the dataset"""
        if self.encoded_ds:
            self.retriever.load_index_from_disk()
            self.ds = self.retriever.get_docs(self.encoded_ds, self.k)
            self.__contains_documents = True
        else:
            raise RuntimeError("Must prepare queries")

    def contains_documents(self):
        """Retruns true if document retrieval has been performed on the dataset"""
        return self.__contains_documents

    def save_document_dataset(self, output_dir: str):
        """Saves dataset with documents to the output directory. Dataset must contain documents."""
        if self.__contains_documents:
            self.ds.save_to_disk(
                output_dir + "/" + str(self.retriever) + "_" + self.dataset_name
            )
        else:
            raise RuntimeError(
                "Docs Not Processed: Must retrieve documents before saving"
            )
