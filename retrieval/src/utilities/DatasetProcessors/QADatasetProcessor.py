from abc import ABC, abstractmethod
from datasets import DatasetDict


class QADatasetProcessor(ABC):
    """
    Abstract class for QA Datasets containing question, answer, and documents"""

    @abstractmethod
    def __init__(self, dataset_path: str):
        """Initializes and loads QA dataset that contains documents.

        Args:
            dataset_path (str): Path to the dataset files.
        """
        pass

    @abstractmethod
    def get(self) -> DatasetDict:
        """Returns processed QA dataset with question, answer, and document fields"""
        pass
