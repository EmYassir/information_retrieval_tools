# A Generative Advaserial Neural Network based Passage Re-Ranking method implemented in Pytorch

## Dependencies:
1. Python: 3.7+
2. Pytroch: 1.6+
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


## Training Procedure:
1. please link the data file paths to your data files in the code before you run the code.
2. run
    python trainranker.py --dataset quasart --batch_size 8 --num_epochs 8 --span_len 5 --default_num_docs 50
3. check the running log and training results in logs and output folder respectively.



