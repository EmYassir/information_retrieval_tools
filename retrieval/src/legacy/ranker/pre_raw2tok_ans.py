'''
Tokenize the data file, saving the cost of the online tokenization
Execute after pre_docrank.py
'''
import io
import json
from tqdm import tqdm
import tokenizers
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
import re

PROCESS_TOK = None

tokenizers.set_default('corenlp_classpath', '/u/pandu/soft/corenlp/*')
tok_class = tokenizers.get_class('corenlp')

def init(tokenizer_class, tokenizer_opts):
    global PROCESS_TOK
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)


def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def tokenize_json(docfile, docfile_tok, empty_ids=None):
    processes = ProcessPool(initializer=init, initargs=(tok_class, {}))
    rest = 0
    with open(docfile, 'r') as f:
        lines = f.read().splitlines()
        with open(docfile_tok, 'w+') as af:
            for lid, line in enumerate(tqdm(lines, desc="Tokenizing Paragras")):
                if empty_ids and lid in empty_ids:
                    continue
                exs = json.loads(line)
                documents = [ex['document'] for ex in exs]
                documents.append(exs[0]['question'])
                a_tokens = processes.map_async(tokenize_text, documents)
                a_tokens = a_tokens.get()
                p_tokens = [tk.words() for tk in a_tokens[:-1]]
                q_tokens = a_tokens[-1].words()
                for idx in range(len(exs)):
                    exs[idx]['question'] = q_tokens
                    exs[idx]['document'] = p_tokens[idx]
                af.write(json.dumps(exs)+'\n')
                rest += 1
            af.flush()

        print(rest, 'of', len(lines), "questions were written to file", docfile_tok)

def answerize_txt(answerfile, docfile_tok, answerfile_tok, is_test=None):
    texts = []
    with open(docfile_tok, 'r') as df:
        lines = df.read().splitlines()
        for line in tqdm(lines, desc="Loading Documents"):
            exs = json.loads(line)
            texts.append([ex['document'] for ex in exs])
    step = 0
    newlines = []
    empty_idx = []
    with open(answerfile, 'r') as f:
        lines = f.read().splitlines()
        for lidx, line in enumerate(tqdm(lines, desc="Extracting Answers")):
            new_ex = {}
            ex = json.loads(line)
            new_ex['question'] = ex['question']
            answer = ex['answer']
            answers = []
            try:
                ans_regex = re.compile("(%s)"%answer[0], flags=re.IGNORECASE + re.UNICODE)

                docs = texts[lidx]
                for text in docs:
                    # paragraph = " ".join(text)
                    paragraph = text
                    par_ans = ans_regex.findall(paragraph)
                    par_anst = []
                    for an in par_ans:
                        if isinstance(an, str) and len(an.strip()) > 0:
                            if len(an.strip()) < 50: # less than 50 characters
                                par_anst.append(an.strip())
                        elif isinstance(an, tuple) and len(an[0].strip()) > 0:
                            if len(an[0].strip()) < 50:
                                par_anst.append(an[0].strip())
                        else:
                            print(lidx, an)
                    if len(par_anst) > 0:
                        answers.extend(par_anst)
            except Exception as exce:
                print(exce)
                print(lidx, answer[0])

            new_ex['answers'] = list(set(answers))
            if len(new_ex['answers']) == 0:
                empty_idx.append(lidx)
            else:
                newlines.append(json.dumps(new_ex))
            step += 1
        print('total', step, 'empty', len(empty_idx), 'rest', len(newlines))
    with io.open(answerfile_tok, 'w+', encoding='utf-8') as af:
        for ln in newlines:
            af.write(ln + '\n')

    return empty_idx


def answerize_txt_trec_test(answerfile, docfile_tok, answerfile_tok):
    texts = []
    with open(docfile_tok, 'r') as df:
        lines = df.read().splitlines()
        for line in tqdm(lines, desc="Loading Documents"):
            exs = json.loads(line)
            texts.append([ex['document'] for ex in exs])
    step = 0
    newlines = []
    empty_idx = []
    with open(answerfile, 'r') as f:
        lines = f.read().splitlines()
        for lidx, line in enumerate(tqdm(lines, desc="Extracting Answers")):
            new_ex = {}
            ex = json.loads(line)
            new_ex['question'] = ex['question']
            answer = ex['answer']
            answers = []
            try:
                ans_regex = re.compile("(%s)" % answer[0], flags=re.IGNORECASE + re.UNICODE)

                docs = texts[lidx]
                for text in docs:
                    # paragraph = " ".join(text)
                    paragraph = text
                    par_ans = ans_regex.findall(paragraph)
                    par_anst = []
                    for an in par_ans:
                        if isinstance(an, str) and len(an.strip()) > 0:
                            # if len(an.strip()) < 50:  # less than 50 characters
                            par_anst.append(an.strip())
                        elif isinstance(an, tuple) and len(an[0].strip()) > 0:
                            for ant in an:
                                if len(ant.strip()) > 0:
                                    par_anst.append(ant.strip())
                        else:
                            print(lidx, an)
                    if len(par_anst) > 0:
                        answers.extend(par_anst)
            except Exception as exce:
                print(exce)
                print(lidx, answer[0])

            new_ex['answers'] = list(set(answers))
            if len(new_ex['answers']) == 0:
                empty_idx.append(lidx)

            newlines.append(json.dumps(new_ex))
            step += 1
        print('total', step, 'empty', len(empty_idx), 'rest', len(newlines))
    with io.open(answerfile_tok, 'w+', encoding='utf-8') as af:
        for ln in newlines:
            af.write(ln + '\n')
    print('empty', len(empty_idx), 'trec test, not providing empty list')
    return []

## ========= raw dataset and its splits ====================================
# dset = 'webquestions'
# dset = 'trec'
# dset = 'trec_d100_p500'
dset = 'trec100_full'
# dset = 'webquestions100_full'
# dset = 'webquestions_d100_p500'

# splt = 'train' # trec_d100_p500 0.93069
splt = 'test' # trec: 0.87752, trec_d100_p500: 0.93948, trec_full: 0.89481/1.0

''' Orig files tokenization and extract answers if necessary (for trec dataset only)'''
pre_fn = 'WebQuestions' if 'webquestions' in dset else 'CuratedTrec'
# dfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+pre_fn + '-' + splt +'-top100-paragraph.jsonl'
dfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+pre_fn + '-' + splt +'-bm25-dtop100-ptop500-paragraph.jsonl'
tdfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'.json'
empty_lids = None
if 'trec' in dset:
    afile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ 'CuratedTrec-' + splt +'.txt'
    exafile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'.txt'
    # extract answers
    if splt == 'test':
        empty_lids = answerize_txt_trec_test(afile, dfile, exafile)
    else:
        empty_lids = answerize_txt(afile, dfile, exafile)
# tokenize documents
tokenize_json(dfile, tdfile, empty_ids=empty_lids)
print('\nDone!')