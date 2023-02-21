import pandas as pd
from datasets import load_dataset, DatasetDict
from src.utilities.DatasetProcessors.OpenQADatasetProcessor import (
    OpenQADatasetProcessor,
)
from src.utilities.Retriever.Retriever import Retriever


class NaturalQuestionsProcessor(OpenQADatasetProcessor):
    """Dataset Processor for NaturalQuestions dataset

    Source: https://github.com/google-research/language/tree/b76d2230156abec5c8d241073cdccbb36f66d1de/language/orqa

    Note: dataset_path points to RESPLIT_PATH
    """

    def __init__(self, dataset_path: str, retriever: Retriever, k=100):
        super().__init__(retriever, k)
        self.dataset_name = "Natural_Questions"
        self.ds_no_docs = self.__load_qa_dataset(dataset_path)

    def __load_qa_dataset(self, dataset_path: str) -> DatasetDict:
        splits = ["train", "test", "dev"]
        DATA_FILES = {
            split: dataset_path + "/NaturalQuestions.resplit." + split + ".jsonl"
            for split in splits
        }
        return load_dataset("json", data_files=DATA_FILES)
