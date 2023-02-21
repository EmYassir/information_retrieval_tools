## Monthly Progress

| Assignee                      | Deadline   | Progress                                                                 | Description                                |
|-------------------------------|------------|--------------------------------------------------------------------------|--------------------------------------------|
| Oussama             | 2022-05-20 | ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/100) | Litrature review  and brainstorming                      |
| Yassir             | 2022-05-20 | ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/100) | Litrature review  and brainstorming                      |

## Research Proposal: Adversarial Answer Oriented Passage Re-Ranking

* **Task** Question Answering

* **Problem statement** QA models can accurately extract the correct answer span but most of the time from an irrelevant Paragraph.  

* **Project Objective**
    * [ ] Run the current baseline on large datasets and improve it with BERT.
    * [ ] Upgrade the GAN framework to mask-edit generative GAN.
    * [ ] Annotate a subset of challenging <contain answer, irrelevant> passages and advocate the evaluation on it.  
     
* **A top-tier conference paper is publishable if**: 
    * [ ] BERT improves the baseline compared to RNN encoder.
    * [ ] Our new GAN approach significantly outperform the improved baseline.
    * [ ] We distribute a dataset with challenging <contain answer, irrelevant> passages, where we show that: current systems perform poorly, our approach works much better.  

### Here is the roadmap of the project [May-September]

* [ ] Implement the baseline [2-3 week]
    * [ ] Run the base code and reproduce the paper results.
    * [ ] Run the code on large scale datasets.
    * [ ] Refactor the code: 
        * [ ] New dataloader.
        * [ ] Dense Retrieval tools (both BM25, DPR\FAISS)
        * [ ] Automatic Fine-tune and evaluate pipeline. 

* [ ] Improve the current baseline [1-2.5 week]
    * [ ] Replace RNN by BERT encoder.
    * [ ] Test different Models type (Splinter, Roberta, SpanBERT) and sizes (small, base, large).
    * [ ] Generate benchmarking results model size vs. latency.

* [ ] Explore 1: Post-Reranking with GAN [2-3 week]
    * [ ] Replace the MRC generator by a re-ranker classifier.
    * [ ] Place the GAN framework after the reader and compare with previous approach.
    * [ ] Goal: significant improvement in term of latency and performance compared to the baseline.

* [ ] Explore 2: Adversarial DA with GAN [3-5 weeks]
    * [ ] Develop methods to select Question-Passage tokens that mostly influence model (heuristic, random, attention).  
    * [ ] Explore the mask-and-edit adversarial  approach (MATE-KD) for getting hard <contain answer-irrelevant> examples.
    * [ ] Explore the mask-and-edit adversarial  approach (MATE-KD) for getting hard <not contain answer-relevant> examples.
    * [ ] (optional) DA methods for hard examples generation (QG).

* [ ] Explore 3: Dataset Building [2.5-3.5 weeks]
    * [ ] Get an off-the-shelf-retrieval and Ranker model.
    * [ ] Predict on dev set of multiple datasets and select the one(s) that contains hard  <contain answer-irrelevant> examples.
    * [ ] Develop a semi-automatic method to filter down the hard <contain answer-irrelevant> examples.
    * [ ] Design an annotation task and assign it to annotator.
    * [ ] Annotate (done by external annotator).
    * [ ] Ensure that the sub select set is challenging for systems.

* **Reference List**
    * PReGAN: Answer Oriented Passage Ranking with Weakly
Supervised GAN
    * [MATE-KD: Masked Adversarial TExt, a Companion to Knowledge
Distillation](https://arxiv.org/pdf/2105.05912.pdf)


## How to run the code (RANKER):
Note that in steps 1-4, you will have to change the hard coded paths to the desired dataset, embedding, and output directory.
0) Download Stanford CoreNLP from https://stanfordnlp.github.io/CoreNLP/
1) Run normdata.py 
2) Run pre_docrank.py
3) Run pre_txt_token.py
4) Run trainranker.py

## How to run the code (Dataset Preprocessing):
1) Download the dataset you wish to use. Instructions for that is under src/utilities/DatasetProcessors/{YOUR_DATASET}Processor.py.
2)
    - Use tools/ProcessQADataset.py for processing QuasarT
    - Use tools/RetrieveDocsOpenQA.py for processing open domain question answering datasets (TriviaQA, Natural Questions, WebQuestions, CuratedTREC)
3) (OPTIONAL)
    - If you wish to tokenize the dataset using the DPRReader style (ie question[SEP]title[SEP]document) with DPR Reader scores, use tokenize_dpr_processed_datasets.py.
    - This script adds a column with DPRReader's relevance logits and boolean values on where DPRReader's predicted answer is correct or not. 
    - Note that the DPRReader only currently uses a single GPU.

NOTE: BM25 retriever, though working, is very slow. Future improvement would be to use the Pyserini retrieval instead.
NOTE 2: CuratedTREC is the only dataset with regex as the answer, and not a list of answer strings. 

### Source for Datasets:
1)  - QuasarT source: http://curtis.ml.cmu.edu/datasets/quasar/
    - dataset_path must point to the quasar-t directory
2)  - NaturalQuestions source: https://github.com/google-research/language/tree/b76d2230156abec5c8d241073cdccbb36f66d1de/language/orqa
    - dataset_path points to RESPLIT_PATH
3)  - TriviaQA source: http://nlp.cs.washington.edu/triviaqa/
    - The dataset_path must point to triviaqa-unflitered directory
4)  - WebQuestions source: https://github.com/google-research/language/tree/b76d2230156abec5c8d241073cdccbb36f66d1de/language/orqa
    - dataset_path points to RESPLIT_PATH
5)  - CuratedTREC source: https://github.com/google-research/language/tree/b76d2230156abec5c8d241073cdccbb36f66d1de/language/orqa
    - dataset_path points to RESPLIT_PATH

## Other useful code:
- the ndcg.py file is a Huggingface metric which handles flattened datasets and can compute hits/ndcg. It does so by unflattening the dataset into its original form.
- train_ans_classifier.py gives an example on how to use the metric found in ndcg.py. Note that this trainer is launched using ```accelerate launch```
- dpr_reader_multi_gpu.py is a utilities file that allows you to run DPRReader on the entire dataset. It must be passed to the multi_gpu_perform function. There is an example of how to use multi_gpu_perform in the DPR.py file.