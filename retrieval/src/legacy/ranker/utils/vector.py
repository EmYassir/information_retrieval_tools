#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Edit from DrQA"""

from collections import Counter
import torch
import logging

logger = logging.getLogger(__name__)

def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return document, features, question, start, end, ex['id']


def vectorize1(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    return document, features, question, ex['id']


def vectorize_with_doc(ex, index, model, num_docs=100, single_answer=False, docs_tmp=None):
    # res = vectorize(ex, model, single_answer)
    # qid = query2id[" ".join(ex['question'])]
    # logger.info("qid=%d", qid)
    # docs_tmp = linecache.getline(filename, qid)
    # docs_tmp = json.loads(docs_tmp)
    docs = []
    for i in range(0, num_docs):
        j = i % len(docs_tmp)
        docs_tmp[j]['answer'] = ex['answer'][0]
        docs_tmp[j]['id'] = index
        # logger.info(ex['answer'][0])
        # logger.info(docs_tmp[j])
        docs.append(vectorize1(docs_tmp[j], model, single_answer))
    return {"qa": ex, "docs": docs}


def vectorize_with_doc2(ex, index, word_dict, feature_dict, num_docs=100, single_answer=False, docs_tmp=None):
    # res = vectorize(ex, model, single_answer)
    # qid = query2id[" ".join(ex['question'])]
    # logger.info("qid=%d", qid)
    # docs_tmp = linecache.getline(filename, qid)
    # docs_tmp = json.loads(docs_tmp)
    docs = []
    for i in range(0, num_docs):
        j = i % len(docs_tmp)
        docs_tmp[j]['answers'] = ex['answers']
        docs_tmp[j]['xid'] = index
        # logger.info(ex['answer'][0])
        # logger.info(docs_tmp[j])
        docs.append(vectorize2(docs_tmp[j], word_dict, feature_dict, single_answer))
    return {"qa": ex, "docs": docs}


def vectorize2(ex, word_dict, feature_dict, single_answer=False):
    """Torchify a single example."""
    use_lemma = True

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    q_words_cased = {w for w in ex['question']}
    q_words_uncased = {w.lower() for w in ex['question']}
    # q_lemma = {w for w in ex['qlemma']} if use_lemma else None
    for i in range(len(ex['document'])):
        if ex['document'][i] in q_words_cased:
            features[i][feature_dict['in_question']] = 1.0
        if ex['document'][i].lower() in q_words_uncased:
            features[i][feature_dict['in_question_uncased']] = 1.0
        # if q_lemma and ex['lemma'][i] in q_lemma:
        #     features[i][feature_dict['in_question_lemma']] = 1.0

    # # f_{token} (POS)
    # for i, w in enumerate(ex['pos']):
    #     f = 'pos=%s' % w
    #     if f in feature_dict:
    #         features[i][feature_dict[f]] = 1.0
    #
    # # f_{token} (NER)
    # for i, w in enumerate(ex['ner']):
    #     f = 'ner=%s' % w
    #     if f in feature_dict:
    #         features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    counter = Counter([w.lower() for w in ex['document']])
    l = len(ex['document'])
    for i, w in enumerate(ex['document']):
        features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    return document, features, question, ex['xid']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x2, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][3]):
            y_s = torch.cat([ex[3] for ex in batch])
            y_e = torch.cat([ex[4] for ex in batch])
        else:
            y_s = [ex[3] for ex in batch]
            y_e = [ex[4] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids


def batchify1(batch, parag_len):
    """Gather a batch of individual examples into one batch."""

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]

    # Batch documents and features
    max_length = min(max([d.size(0) for d in docs]), parag_len) # maximum length is parag_len
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        clen = min(d.size(0), max_length)
        x1[i, :clen].copy_(d[:clen])
        x1_mask[i, :clen].fill_(0)
        if x1_f is not None:
            x1_f[i, :clen].copy_(features[i][:clen])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
    # print(ids)
    return x1, x1_f, x1_mask, x2, x2_mask, ids


def batchify_with_docs(batch_list, num_docs, parag_len=150):
    res = []
    for i in range(num_docs):
        batch = []
        for ex in batch_list:
            batch.append(ex['docs'][i])
        res.append(batchify1(batch, parag_len))
    # logger.info("batchify_with_docs%d", len(res))
    return res
