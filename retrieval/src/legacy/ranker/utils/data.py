# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Edit from DrQA"""
import numpy as np
import logging
import unicodedata
import time
import json
import io
import re

from collections import Counter
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.sampler import Sampler
from utils.vector import vectorize, vectorize_with_doc2
from tqdm import tqdm

import tokenizers
import socket
from multiprocessing.util import Finalize

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# TOK
host = socket.gethostname()
print('Running on host', host)
if host == 'Du':
    tokenizers.set_default('corenlp_classpath', '/Users/pan/Workshop/pyspace/corenlp/stanford-corenlp-full-2018-10-05/*')
elif host == 'mutux':
    tokenizers.set_default('corenlp_classpath', '/Users/pan/Workshop/pyspace/corenlp/stanford-corenlp-full-2018-10-05/*')
else:
    tokenizers.set_default('corenlp_classpath', '/u/pandu/soft/corenlp/*')
global PROCESS_TOK
tok_class = tokenizers.get_class("corenlp")
tok_opts = {}
PROCESS_TOK = tok_class(**tok_opts)
Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class ReaderDataset(Dataset):

    def __init__(self, examples, model, single_answer=False):
        self.model = model
        self.examples = examples
        self.single_answer = single_answer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        return vectorize(self.examples[index], self.model, self.single_answer)

    def lengths(self):
        return [(len(ex['document']), len(ex['question']))
                for ex in self.examples]


def has_answer(answer, t, dataset='quasart'):
    global PROCESS_TOK
    text = []
    for i in range(len(t)):
        text.append(t[i].lower())
    res_list = []
    if dataset == "trec":
        try:
            ans_regex = re.compile("(%s)"%answer[0], flags=re.IGNORECASE + re.UNICODE)
        except:
            return False, res_list
        paragraph = " ".join(text)
        answer_new = ans_regex.findall(paragraph)
        for a in answer_new:
            single_answer = normalize(a[0])
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    res_list.append((i, i+len(single_answer)-1))
    else:
        for a in answer:
            # single_answer = " ".join(a).lower()
            single_answer = a
            single_answer = normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    res_list.append((i, i+len(single_answer)-1))

    if (len(res_list)>0):
        return True, res_list
    else:
        return False, res_list


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


class ReaderDataset_with_Doc(Dataset):

    def __init__(self, examples, docs, word_dict, feature_dict, num_docs=100, single_answer=False):
        self.examples = examples
        self.single_answer = single_answer
        self.docs = docs
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.num_docs = num_docs
        #for i in range(len(self.examples)):
        #    for j in range(0, len(self.docs_by_question[i])):
        #        self.docs_by_question[i]['has_answer'] = has_answer(self.examples[i]['answer'], self.docs_by_question[i][document])
        #print (self.docs_by_question.keys())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        question = self.examples[index]['question']
        #logger.info("%d\t%s",index, question)
        #logger.info(self.docs_by_question[question])
        #assert("\n" not in question)
        #if (question not in self.docs_by_question):
        #    logger.info("No find question:%s", question)
        #    return []
        return vectorize_with_doc2(self.examples[index], index, self.word_dict, self.feature_dict, self.num_docs, self.single_answer, self.docs[index])

    def lengths(self):
        #return [(len(ex['document']), len(ex['question']))
        #        for ex in self.examples]
        return [(len(doc[self.num_docs-1]['document']), len(doc[self.num_docs-1]['question'])) for doc in self.docs]


class ReaderIterableDataset_with_Doc(IterableDataset):

    def __init__(self, q_ans_file, q_docs_file, word_dict, feature_dict, tot_len, num_docs=100, single_answer=False):
        self.q_ans_file = q_ans_file
        self.single_answer = single_answer
        self.q_docs_file = q_docs_file
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.num_docs = num_docs
        self.ind_iter = iter(range(tot_len))
        #for i in range(len(self.examples)):
        #    for j in range(0, len(self.docs_by_question[i])):
        #        self.docs_by_question[i]['has_answer'] = has_answer(self.examples[i]['answer'], self.docs_by_question[i][document])
        #print (self.docs_by_question.keys())
    def process_doc(self, q_docs):
        ex = json.loads(q_docs, encoding='utf-8')
        # logger.info(ex)
        # print (ex)

        try:
            question = " ".join(ex[0]['question'])
        except:
            logger.info(ex)
        # print (question)
        for i in range(len(ex)):
            # ex_que = norm_lst(ex[i]['question'])
            ex_que = ex[i]['question']
            ex[i]['question'] = [w.lower() for w in ex_que]
            # ex_doc = norm_lst(ex[i]['document'])
            ex_doc = ex[i]['document']
            ex[i]['document'] = [w.lower() for w in ex_doc]
        # logger.info (question)
        tmp_res = []
        for i in range(len(ex)):
            if len(ex[i]['document']) > 2:  # and len(ex[i]['document'])<200):
                tmp_res.append(ex[i])
            if len(tmp_res) >= self.num_docs:
                break
        # maintaining original index, so that spans and labels can be indexed by it
        tmp_res_ind_map = list(range(len(tmp_res)))

        if len(tmp_res) == 0:
            print(ex)

        if len(tmp_res) < self.num_docs:
            len_tmp_res = len(tmp_res)
            for i in range(len_tmp_res, self.num_docs):
                tmp_res.append(tmp_res[i - len_tmp_res])
                tmp_res_ind_map.append(tmp_res_ind_map[i - len_tmp_res])
        ind_tmp_sorted = sorted(enumerate(tmp_res), key=lambda x: len(x[1]['document']))
        tmp_res = [r[1] for r in ind_tmp_sorted]
        tmp_ind = [tmp_res_ind_map[r[0]] for r in ind_tmp_sorted]
        assert (len(tmp_res) != 0)
        # if (len(tmp_res)!=num_docs):
        #    logger.info("%s\t%d\t%d", question, len(tmp_res),num_docs)
        return tmp_res, question, tmp_ind

    def process_ans(self, q_ans):
        # only works for quasar-t format
        data = json.loads(q_ans)
        answer = [tokenize_text(a).words() for a in data['answers']]
        question = " ".join(tokenize_text(data['question']).words())
        return {"answers": answer, "question": question, "answers_orig": data['answers']}

    def process(self, q_ans, q_docs, index):
        dev_docs, dev_questions, dev_docs_inds = self.process_doc(q_docs)
        dev_exs_with_doc = self.process_ans(q_ans)
        return vectorize_with_doc2(dev_exs_with_doc, index, self.word_dict, self.feature_dict, self.num_docs,
                            self.single_answer, dev_docs), dev_exs_with_doc, dev_docs

    def __iter__(self):
        q_docs_iter = io.open(self.q_docs_file,'r', encoding='utf-8')
        q_ans_iter = io.open(self.q_ans_file, 'r', encoding='utf-8')
        data_iter = map(self.process, q_ans_iter, q_docs_iter, self.ind_iter)

        return data_iter

# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_data_with_doc_with_index_4_1_batch(b_id, exs, docs, indices, batch_size):
    docs_batch = docs[b_id * batch_size: (b_id + 1) * batch_size]
    exs_batch_neg = exs[b_id * batch_size: (b_id + 1) * batch_size]
    train_docs_neg = []
    for question_ind_nd in range(len(indices)):
        que = []
        question_ind = indices[question_ind_nd].tolist()
        for doc_ind in question_ind:
            que.append(docs_batch[question_ind_nd][doc_ind])
        train_docs_neg.append(que)
    return exs_batch_neg, train_docs_neg


def load_data_with_doc(filename, num_docs):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    res_ind = []
    res = []
    keys = set()
    step =0
    # tot_len = sum(1 for i in open(filename, 'r'))
    with io.open(filename,'r', encoding='utf-8') as f:
        # for line in tqdm(f,total=tot_len):
        for line in f:
            ex = json.loads(line, encoding='utf-8')
            #logger.info(ex)
            #print (ex)
            step+=1
            try:
                question = " ".join(ex[0]['question'])
            except:
                logger.info(step)
                logger.info(ex)
                continue
            #print (question)
            for i in range(len(ex)):
                # ex_que = norm_lst(ex[i]['question'])
                ex_que = ex[i]['question']
                ex[i]['question'] = [w.lower() for w in ex_que]
                # ex_doc = norm_lst(ex[i]['document'])
                ex_doc = ex[i]['document']
                ex[i]['document'] = [w.lower() for w in ex_doc]
            #logger.info (question)
            tmp_res = []
            for i in range(len(ex)):
                if len(ex[i]['document'])>2:# and len(ex[i]['document'])<200):
                    tmp_res.append(ex[i])
                if len(tmp_res)>=num_docs:
                    break
            # maintaining original index, so that spans and labels can be indexed by it
            tmp_res_ind_map = list(range(len(tmp_res)))

            if len(tmp_res) == 0:
                print(ex)
                continue

            if len(tmp_res)<num_docs:
                len_tmp_res = len(tmp_res)
                for i in range(len_tmp_res, num_docs):
                    tmp_res.append(tmp_res[i-len_tmp_res])
                    tmp_res_ind_map.append(tmp_res_ind_map[i-len_tmp_res])
            ind_tmp_sorted = sorted(enumerate(tmp_res), key=lambda x:len(x[1]['document']))
            tmp_res = [r[1] for r in ind_tmp_sorted]
            tmp_ind = [tmp_res_ind_map[r[0]] for r in ind_tmp_sorted]
            assert(len(tmp_res)!=0)
            #if (len(tmp_res)!=num_docs):
            #    logger.info("%s\t%d\t%d", question, len(tmp_res),num_docs)
            res.append(tmp_res)
            keys.add(question)
            res_ind.append(tmp_ind)
    return res, keys, res_ind


def read_data(filename, keys):
    res = []
    step = 0
    for line in open(filename):
        data = json.loads(line)
        if ('squad' in filename or 'webquestions' in filename):
            answer = [tokenize_text(a).words() for a in data['answer']]
        else:
            if ('CuratedTrec' in filename):
                answer = data['answer']
            else:
                answer = [tokenize_text(a).words() for a in data['answers']]
        question = " ".join(tokenize_text(data['question']).words())
        res.append({"answers":answer, "question":question, "answers_orig": data['answers']})
        step+=1
    return res


def load_list_with_doc(data_lst, num_docs, pred=False):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    res_ind = []
    res = []
    keys = set()
    step =0
    for ex in data_lst:
        step+=1
        try:
            question = " ".join(ex[0]['question'])
        except:
            logger.info(step)
            logger.info(ex)
            continue
        tmp_res = []
        for i in range(len(ex)):
            if len(ex[i]['document'])>2:# and len(ex[i]['document'])<200):
                tmp_res.append(ex[i])
            if len(tmp_res)>=num_docs:
                break
        if len(tmp_res) == 0:
            continue
        # maintaining original index, so that spans and labels can be indexed by it
        tmp_res_ind_map = list(range(len(tmp_res)))
        if len(tmp_res) == 0:
            print(ex)
        if len(tmp_res)<num_docs:
            len_tmp_res = len(tmp_res)
            for i in range(len_tmp_res, num_docs):
                tmp_res.append(tmp_res[i-len_tmp_res])
                tmp_res_ind_map.append(tmp_res_ind_map[i-len_tmp_res])
        if pred:
            res.append(tmp_res)
            keys.add(question)
            res_ind.append(tmp_res_ind_map)
        else:
            ind_tmp_sorted = sorted(enumerate(tmp_res), key=lambda x:len(x[1]['document']))
            tmp_res = [r[1] for r in ind_tmp_sorted]
            tmp_ind = [tmp_res_ind_map[r[0]] for r in ind_tmp_sorted]
            assert(len(tmp_res)!=0)
            #if (len(tmp_res)!=num_docs):
            #    logger.info("%s\t%d\t%d", question, len(tmp_res),num_docs)
            res.append(tmp_res)
            keys.add(question)
            res_ind.append(tmp_ind)
    return res, keys, res_ind


def read_list(data_lst, args):
    res = []
    step = 0
    for data in data_lst:
        if ('squad' in args.dataset or 'webquestions' in args.dataset):
            answer = [tokenize_text(a).words() for a in data['answer']]
        else:
            if ('CuratedTrec' in args.dataset):
                answer = data['answer']
            else:
                answer = [tokenize_text(a).words() for a in data['answers']]
        question = " ".join(tokenize_text(data['question']).words())
        res.append({"answers":answer, "question":question})
        step+=1
    return res


def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def build_feature_dict():#, examples):
    """Index features (one hot) from fields in examples and options."""
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    # Exact match features
    _insert('in_question')
    _insert('in_question_uncased')
    _insert('in_question_lemma')
    '''
    # Part of speech tag features
    if args.use_pos:
        for ex in examples:
            for w in ex['pos']:
                _insert('pos=%s' % w)

    # Named entity tag features
    if args.use_ner:
        for ex in examples:
            for w in ex['ner']:
                _insert('ner=%s' % w)
    '''
    # Term frequency feature
    _insert('tf')
    return feature_dict


def build_word_dict_docs(docs, restrict_vocab=False, embedding_file=None):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words_with_docs(docs, restrict_vocab, embedding_file):
        word_dict.add(w)
    return word_dict


def load_words_with_docs(docs, restrict_vocab=False, embedding_file=None):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if restrict_vocab and embedding_file:
        logger.info('Restricting to words in %s' % embedding_file)
        valid_words = index_embedding_words(embedding_file, 500000)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for examples in docs:
        for ex in examples:
            _insert(ex['question'])
            _insert(ex['document'])
    return words

def load_data(args, filename, skip_no_answer=False):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with io.open(filename, 'r', encoding='utf-8') as f:
        examples = [json.loads(line, encoding='utf-8') for line in f]

    # Make case insensitive?
    if args.uncased_question or args.uncased_doc:
        for ex in examples:
            if args.uncased_question:
                ex['question'] = [w.lower() for w in ex['question']]
            if args.uncased_doc:
                ex['document'] = [w.lower() for w in ex['document']]

    # Skip unparsed (start/end) examples
    if skip_no_answer:
        examples = [ex for ex in examples if len(ex['answers']) > 0]

    return examples


def load_text(filename):
    """Load the paragraphs only of a SQuAD dataset. Store as qid -> text."""
    # Load JSON file
    with io.open(filename, 'r', encoding='utf-8') as f:
        examples = json.load(f)['data']

    texts = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                texts[qa['id']] = paragraph['context']
    return texts


def load_answers(filename):
    """Load the answers only of a SQuAD dataset. Store as qid -> [answers]."""
    # Load JSON file
    with io.open(filename, 'r', encoding='utf-8') as f:
        examples = json.load(f)['data']

    ans = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                ans[qa['id']] = list(map(lambda x: x['text'], qa['answers']))
    return ans


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------
def index_embedding_words(embedding_file, num_words=None):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in f:
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
            if (num_words is not None and len(words) >= num_words):
                break
    return words


def load_words(args, examples):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if args.restrict_vocab and args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        valid_words = index_embedding_words(args.embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for ex in examples:
        _insert(ex['question'])
        _insert(ex['document'])
    return words


def build_word_dict(args, examples):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict


def top_question_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['question']:
            w = Dictionary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)

# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
