from typing import Optional
import pandas as pd
from datasets import Dataset, DatasetDict
from src.utilities.DatasetProcessors.OpenQADatasetProcessor import (
    OpenQADatasetProcessor,
)

from src.utilities.Retriever.Retriever import Retriever


class TriviaQAProcessor(OpenQADatasetProcessor):
    """
    Preprocessor for the TriviaQA Dataset (Unfiltered)
    Source for dataset: http://nlp.cs.washington.edu/triviaqa/

    Note: The dataset_path must point to triviaqa-unflitered directory
    Note: The test dataset does not contain answers
    """

    def __init__(self, dataset_path: str, retriever: Retriever, k=100):
        super().__init__(retriever, k)
        self.dataset_name = "TriviaQA"
        self.ds_no_docs = self.__load_qa_dataset(dataset_path)

    def __load_qa_dataset(self, dataset_path):
        train = self.__process_split(dataset_path, "train")
        dev = self.__process_split(dataset_path, "dev")
        test = self.__process_split(
            dataset_path, "test-without-answers", with_answers=False
        )
        return DatasetDict({"train": train, "dev": dev, "test": test})

    def __process_split(
        self, path: str, split: str, with_answers: Optional[bool] = True
    ):
        trivia_dev = pd.read_json(path + "/unfiltered-web-" + split + ".json")
        if with_answers:
            trivia_dev["answer"] = trivia_dev["Data"].map(
                lambda example: example["Answer"]["Aliases"]
            )
        trivia_dev["question"] = trivia_dev["Data"].map(
            lambda example: example["Question"]
        )
        trivia_dev = trivia_dev.drop(
            ["Version", "Data", "Domain", "VerifiedEval", "Split"], axis=1
        )
        return Dataset.from_pandas(trivia_dev, split=split, preserve_index=False)
