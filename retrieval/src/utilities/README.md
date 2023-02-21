# Overview
The packagte qa_preprocess has two types of modules: Retrievers and DatasetProcessors.
## Retreiver
Constructs an index to perform retrieval on a given corpus. Used by a dataset processor to retrieve documents for an OpenQADataset. Two types of retriever are defined:
1) DPR: Constructs a FAISS index to perform retrieval.
2) BM25: Tokenizes passages by word and uses rank-bm25 to retrieve.

### To contruct an index:
1) Load corpus with a text column and create an instance of a Retreiver
2) Run build_index_for_corpus
## Dataset Processors
Dataset processors read raw datasets and convert them to a unified format containing question, answers, and documents. 
- QADatasetProcessor is an abstract class for dataset processors that output datasets with documents already contained. 
- OpenQADatasetProcessor is a superclass for datasets that only contain questions and answers. To convert to the unified format, OpenQADatasetProcessors take a Retriever to retrieve documents from a corpus. It then outputs a dataset with questions, answers, and k documents, where k is a parameter that can be set.

### Converting OpenQADataset to QADataset
To retrieve documents for an OpenQA Dataset, use the RetrieveDocsOpenQA.py script with the appropriate arguments. The retreiver must be initialized before processing.
## Datasets
We use the following paths to source our datasets

#### WebQuestions, NaturalQuestions, CuratedTREC:
    https://github.com/google-research/language/blob/master/language/orqa/README.md
    To get the WebQuestions, NaturalQuestions, and CuratedTREC datasets, follow the tutorial in the above link. Set the datasetpath for the processors to 
    the RESPLIT_PATH.
#### QuasarT
    https://github.com/bdhingra/quasar
    Set the datasetpath to the quasar_t directory.
#### TriviaQA
    https://nlp.cs.washington.edu/triviaqa/
    Download the unfilitered dataset and set the dataset_path to the triviaqa-unfiltered directory after unzipping.
#### SearchQA
    https://github.com/nyu-dl/dl4ir-searchQA
    UNIMPLEMENTED

## Corpus
The corpus used is the DPR wikipedia corpus found at: https://github.com/facebookresearch/DPR

The corpus filename is psgs_w100.tsv