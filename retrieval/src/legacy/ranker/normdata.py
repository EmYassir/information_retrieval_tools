import tokenizers
import socket
import io
import json
import re
from tqdm import tqdm

def norm_data_with_doc(filename, ofile):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    uni_pat = re.compile('^u([0-9a-f]){4}.*')
    res = []
    res_ind = []
    keys = set()
    step =0
    cnt = 0
    newlines = []
    num_lines = sum(1 for line in open(filename))
    with open(filename, 'r') as f:
        for line in tqdm(f, total=num_lines):
            tmp_res = []
            exs = json.loads(line)
            for ex in exs:
                doc = ex['document']
                doc = norm_lst(doc)
                doc = uni_str(doc, uni_pat)
                ex['document'] = doc

                que = ex['question']
                # que = norm_lst(que)
                que = uni_str(que, uni_pat)
                ex['question'] = que
            newlines.append(json.dumps(exs))
            step+=1
        print('total', step)
    with io.open(ofile, 'w+', encoding='utf-8') as of:
        for ln in newlines:
            of.write((ln + '\n'))

def trash_lst(doc):
    grn = []
    ext = ['meta', 'body', '/', 'a', '!', 'th', 'td', 'font', 'gc3', 'ref', '?', 'tr', 'input', 'div', 'table', 'rdf', 'img', 'src', 'alt', 'p', 'form', 'h2', 'h1', 'h1']
    for i, word in enumerate(doc):
        if '<' in word.lower() and not '<<' in word.lower():
            if i < len(doc) - 1 and any(e in doc[i + 1].lower() for e in ext):
                grn.append(word)
                continue
        if len(grn) > 0:
            if '>' not in word.lower():
                grn.append(word)
                continue
            else:
                grn.append(word)
                return grn

        if i == len(doc) - 1 and '>' in word.lower():
            grn.append(word)
            return grn

    if len(grn) > 0:
        return grn

def norm_lst(doc):
    trash = trash_lst(doc)
    regular = []
    if trash is not None:
        # print(trash)
        for word in doc:
            if word in trash:
                continue
            else:
                regular.append(word)
    else:
        regular = doc
    regl = []
    ext = ['http', 'www.', '<a>', '</a>', '<b>', '</b>', '</font>', '<font>', '<i>', '<!', '</i>', '</math>', '</ref>', '<ref>', '<?xml']
    for word in regular:
        if any(word.startswith(e) for e in ext) or len(word) > 40:
            continue
        else:
            regl.append(word)
    return regl

def uni_str(doc, uni_pat):
    uni_doc = []
    for i, word in enumerate(doc):
        if word.strip().endswith("\\") and i < len(doc) - 1 and re.match(uni_pat, doc[i+1].strip()):
            print(word, doc[i + 1])
            continue
        if re.match(uni_pat, word.strip()) and i > 0 and doc[i - 1].strip().endswith("\\"):
            uni_doc.append((doc[i-1]+word).encode('utf-8').decode('unicode-escape'))
        else:
            uni_doc.append(word)
    return uni_doc

def norm_data(filename, ofile):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    step =0
    newlines = []
    num_lines = sum(1 for line in open(filename))
    with open(filename, 'r') as f:
        for line in tqdm(f, total=num_lines):
            tmp_res = []
            ex = json.loads(line)
            que = ex['question'].replace('\"', '\'').encode('utf-8').decode('unicode-escape')
            ex['question'] = que
            if 'answers' in ex:
                anskey = 'answers'
            elif 'answer' in ex and isinstance(ex['answer'], list):
                anskey = 'answer'
            else:
                anskey = None
            try:
                ans = [an.replace('\"', '\'').encode('utf-8').decode('unicode-escape') for an in ex[anskey]]
            except Exception as excpt:
                ans = [an for an in ex[anskey]]
                print(excpt)
            ex['answers'] = ans
            newlines.append(json.dumps(ex))
            step+=1
        print('total', step)
    with io.open(ofile, 'w+', encoding='utf-8') as of:
        for ln in newlines:
            of.write((ln + '\n').encode('utf-8').decode('unicode-escape'))

# dset = 'nqsub'
# dset = 'unftriviaqa'
# dset = 'searchqa'
# dset = 'quasart'
# dset = 'webquestions'
dset = 'trec'
dset = 'trec_d100_p500'
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

dfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'.json'
udfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm.json'
norm_data_with_doc(dfile, udfile)

qfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'.txt'
uqfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm.txt'
norm_data(qfile, uqfile)