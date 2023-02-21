# A Generative Advaserial Neural Network based Passage Re-Ranking method implemented in Pytorch

## Dependencies:
1. Python: 3.9+
2. Pytroch: 1.10+
3. Tensorboard


## Training data format:
### train.json:
```
[
  {
    "question": [
      "Lockjaw",
      "is",
      "another",
      "name",
      "for",
      "which",
      "disease"
    ],
    "document": [
      "Another",
      "name",
      "for",
      "tetanus",
      "is",
      "lockjaw",
      "."
    ],
    "id": [
      "s3q1674",
      0
    ],
    "has_answers": [
      true,
      [
        [
          3,
          3
        ]
      ]
    ],
    "answers": [
      "tetanus"
    ]
  },
  {
  .
  .
  .
  },
  .
  .
  .
]
```

### train.txt:
```
{
  "question": "Lockjaw is another name for which disease",
  "question_tok": [
    "lockjaw",
    "is",
    "another",
    "name",
    "for",
    "which",
    "disease"
  ],
  "answers": [
    "tetanus"
  ],
  "answers_tok": [
    [
      "tetanus"
    ]
  ]
}
```

## Dataset:
Dataset Supported (All the dataset are publicly available):
1. Quasart-T,
2. TriviaQa,
3. Natural Questions,
4. CuratedTrec,
5. SearchQA
6. For new datasets, you may adpat yours into files in above formats, and apply your algorithm onto them.
Quasar-T, SearchQA, CuratedTrec and TrivialQA: https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenQA_data.tar.gz
Natural Questions: https://github.com/google-research-datasets/natural-questions/tree/master/nq_open

## Run Training and Testing Procedures:
Look into the the two folders for details


