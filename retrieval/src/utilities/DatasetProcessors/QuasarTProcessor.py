import pandas as pd
from datasets import Dataset, DatasetDict
from src.utilities.DatasetProcessors.QADatasetProcessor import QADatasetProcessor


class QuasarTProcessor(QADatasetProcessor):
    """
    DatasetProcessor for QuasarT dataset

    Source: http://curtis.ml.cmu.edu/datasets/quasar/
    Note: dataset_path must point to the quasar-t directory"""

    def __init__(self, dataset_path: str):
        train = self.__process_split(dataset_path, "train")
        dev = self.__process_split(dataset_path, "dev")
        test = self.__process_split(dataset_path, "test")

        self.ds = DatasetDict({"train": train, "dev": dev, "test": test})

    def __process_split(self, dataset_path: str, split: str):
        contexts = pd.read_json(
            dataset_path + "/contexts/short/" + split + "_contexts.json", lines=True
        )
        questions = pd.read_json(
            dataset_path + "/questions/" + split + "_questions.json", lines=True
        )
        ds = pd.merge(questions, contexts, how="outer").astype(
            {"question": str, "answer": str}
        )
        ds["documents"] = ds["contexts"].map(
            lambda passages: [passage[1] for passage in passages]
        )
        ds = Dataset.from_pandas(
            ds[["question", "answer", "uid", "documents"]],
            split="train",
            preserve_index=False,
        )
        ds = ds.map(lambda example: {"answer": [example["answer"]]})

        ds = ds.filter(lambda example: len(example["documents"]) == 100)
        return ds

    def get(self) -> DatasetDict:
        return self.ds
