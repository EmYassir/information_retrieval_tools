'''
Tokenize the data file, saving the cost of the online tokenization
Execute after pre_docrank.py
'''
import io
import json
# from rank_bm25 import BM25Okapi
from utils.data import *
from tqdm import tqdm
from eval import accranker
from utils import data

def tokenize_txt(answerfile, answerfile_tok):
    step = 0
    newlines = []
    with open(answerfile, 'r') as f:
        for line in f:
            new_ex = {}
            if "\\" in line:
                nline = line.replace("\\", "\\\\")
            else:
                nline = line
            ex = json.loads(nline.strip())
            answers = ex['answers']
            answers_tok = []
            for a in answers:
                # single_answer = " ".join(a).lower()
                single_answer = a
                single_answer = normalize(single_answer)
                single_answer = PROCESS_TOK.tokenize(single_answer)
                single_answer = single_answer.words(uncased=True)
                answers_tok.append(single_answer)
            question = ex['question']
            question_tok = normalize(question)
            question_tok = PROCESS_TOK.tokenize(question_tok)
            question_tok = question_tok.words(uncased=True)
            new_ex['question'] = question
            new_ex['question_tok'] = question_tok
            new_ex['answers'] = answers
            new_ex['answers_tok'] = answers_tok
            newlines.append(json.dumps(new_ex))
            step += 1
        print('total', step)
    with io.open(answerfile_tok, 'w+', encoding='utf-8') as af:
        for ln in newlines:
            af.write(ln + '\n')
        print('\nDone!')


def tokenize_json(docfile, docfile_tok):
    step = 0
    newlines = []
    lines = []
    with open(docfile, 'r') as f:
        lines = f.read().splitlines()
    for line in tqdm(lines, total=len(lines)):
        exs = json.loads(line)
        newline = []
        for ex in exs:
            answers = ex['answers']
            answers_tok = []
            for a in answers:
                # single_answer = " ".join(a).lower()
                single_answer = a
                single_answer = normalize(single_answer)
                single_answer = PROCESS_TOK.tokenize(single_answer)
                single_answer = single_answer.words(uncased=True)
                answers_tok.append(single_answer)
            ex['answers_tok'] = answers_tok
            newline.append(ex)
        newlines.append(json.dumps(newline))
        step += 1
    print('total', step)
    print("Writing file to", docfile_tok)
    with io.open(docfile_tok, 'w+', encoding='utf-8') as af:
        for ln in newlines:
            af.write(ln + '\n')
        print('Success!')

# ============ quasart ===========================
''' Answer files tokenziation '''
# afile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm.txt'
# tafile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm.tok.txt'

# afile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm.txt'
# tafile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm.tok.txt'

# afile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm.txt'
# tafile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm.tok.txt'

# afile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm.txt'
# tafile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm.tok.txt'

# afile = 'resources/debug.txt'
# tafile = 'resources/debug.tok.txt'

# tokenize_txt(afile, tafile)

''' Orig files tokenization '''
# dfile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm_orig.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm_tok_orig.json'

# dfile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm_orig.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm_tok_orig.json'

# dfile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm_orig.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm_tok_orig.json'

# dfile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm_orig.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm_tok_orig.json'

# dfile = 'resources/debug_orig.json'
# tdfile = 'resources/debug_tok_orig.json'

''' AQA Rank files tokenization '''
# dfile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm_rank.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm_tok_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm_rank.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm_tok_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm_rank.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm_tok_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm_rank.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm_tok_rank.json'

# dfile = 'resources/debug_rank.json'
# tdfile = 'resources/debug_tok_rank.json'
#

#================== searchqa ==========================
''' Answer files tokenziation '''
# afile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm.txt'
# tafile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm.tok.txt'

# afile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm.txt'
# tafile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm.tok.txt'

# afile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm.txt'
# tafile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm.tok.txt'

# afile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm.txt'
# tafile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm.tok.txt'

# afile = 'resources/debug.txt'
# tafile = 'resources/debug.tok.txt'

# tokenize_txt(afile, tafile)

''' Orig files tokenization '''
# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm_orig.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm_tok_orig.json'

# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm_orig.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm_tok_orig.json'

# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm_orig.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm_tok_orig.json'

# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm_orig.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm_tok_orig.json'

# dfile = 'resources/debug_orig.json'
# tdfile = 'resources/debug_tok_orig.json'

''' AQA Rank files tokenization '''
# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm_rank.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm_tok_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm_rank.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm_tok_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm_rank.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm_tok_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm_rank.json'
# tdfile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm_tok_rank.json'

# dfile = 'resources/debug_rank.json'
# tdfile = 'resources/debug_tok_rank.json'
#

## ========= nqsub ====================================
# dset = 'nqsub'
# dset = 'unftriviaqa'
# dset = 'searchqa'
# dset = 'quasart'
# dset = 'webquestions'
dset = 'trec'
# dset = 'webquestions100'
# dset = 'trec100'
dset = 'trec100_full'
dset = 'webquestions100_full'
dset = 'webquestions_d100_p500'
dset = 'unftriviaqa100'
dset = 'trecqa_as_abl'
dset = 'wqsub'
dset = 'nqsub200'
dset = 'nqsub50'

splt = 'train'
# splt = 'dev'
# splt = 'test'
# splt = 'debug'

''' Answer files tokenziation '''
afile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm.txt'
tafile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm.tok.txt'
tokenize_txt(afile, tafile)

''' Orig files tokenization '''
dfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm_orig.json'
tdfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm_tok_orig.json'

tokenize_json(dfile, tdfile)
