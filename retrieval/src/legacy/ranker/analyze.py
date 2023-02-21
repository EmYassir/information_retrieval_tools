
import json
from tqdm import tqdm
from eval import accranker
from utils import data_tok as data

def paragraph_len_stat(docfile, sfile):
    doclen = {}
    docper = {}
    with open(docfile, 'r') as df:
        lines = df.read().splitlines()
        for line in tqdm(lines):
            exs = json.loads(line)
            for i, ex in enumerate(exs):
                if len(ex['document']) in doclen:
                    doclen[len(ex['document'])] += 1
                else:
                    doclen[len(ex['document'])] = 1
    sorted_len = sorted(doclen.items(), key=lambda kv: kv[0])
    tot = sum([v for k,v in sorted_len])
    percent_len = []
    tv = 0
    f95 = -1
    f96 = -1
    f97 = -1
    f98 = -1
    f99 = -1
    f995 = -1
    f999 = -1
    f100 = sorted_len[-1][0]
    for i, (k, v) in enumerate(sorted_len):
        if i == 0:
            tv = v
        else:
            tv += v
        per = tv/tot
        percent_len.append((k, float("{:.3f}".format(tv/tot))))
        if per > 0.95 and f95 < 0:
            f95 = k
        if per > 0.96 and f96 < 0:
            f96 = k
        if per > 0.97 and f97 < 0:
            f97 = k
        if per > 0.98 and f98 < 0:
            f98 = k
        if per > 0.99 and f99 < 0:
            f99 = k
        if per > 0.995 and f995 < 0:
            f995 = k
        if per > 0.999 and f999 < 0:
            f999 = k
    thr_len = {'p95': f95, 'p96': f96, 'p97': f97, 'p98': f98, 'p99': f99, 'p995': f995, 'p999': f999, 'p100':f100}
    print('Writing stats to file', sfile)
    avg_paragraph_len(sorted_len)
    with open(sfile, 'w+') as sf:
        sf.write(json.dumps(sorted_len) + '\n')
        sf.write(json.dumps(percent_len) + '\n')
        sf.write(json.dumps(thr_len) + '\n')
        sf.flush()
    print(sorted_len)
    print(percent_len)
    print(thr_len)
    print('Done!')

def avg_paragraph_len(sdata):
    num_doc = 0
    tot_len = 0
    lens = None
    if isinstance(sdata, list):
        lens = sdata
    else:
        with open(sdata, 'r') as df:
            line_lens = df.read().splitlines()[0]
            lens = json.loads(line_lens)

    for pair in tqdm(lens):
        k, v = pair
        if k < 5:
            continue
        num_doc += v
        tot_len += k*v
    print(float(tot_len)/float(num_doc))

    print('Done!')

def answer_rate(dfile, afile):
    max_len = 150
    exact_matchs = [data.AverageMeter() for i in range(max_len)]
    exact_matchs_2 = [data.AverageMeter() for i in range(max_len)]

    answers = []
    with open(afile, 'r') as af:
        for line in af:
            axs = json.loads(line)
            # if "\\" in line:
            #     nline = line.replace("\\", "\\\\")
            # else:
            #     nline = line
            # axs = json.loads(nline.strip())
            answers.append(axs["answers"])

    with open(dfile, 'r') as df:
        lines = df.read().splitlines()
        for idx_q, line in enumerate(tqdm(lines)):
            exs = json.loads(line)
            predictions = []

            for idx_doc in range(len(exs)):
                doc_text = exs[idx_doc]["document"]
                predictions.append(" ".join(doc_text))


            if isinstance(answers[idx_q], list):
                answer = answers[idx_q]
            else:
                answer = [answers[idx_q]]
            ground_truths = answer

            mtsi = accranker.topk_metric_max_over_ground_truths(accranker.topk_scope_match_score, predictions,
                                                                ground_truths, max_len)
            for j in range(max_len):
                exact_matchs[j].update(mtsi[j])
            if sum(mtsi) == 0:
                continue
            else:
                for j in range(max_len):
                    exact_matchs_2[j].update(mtsi[j])

        print('Eval:'
              + '\n      E1 = %.4f | E3 = %.4f | E5 = %.4f | E10 = %.4f | E20 = %.4f | E50 = %.4f | E100 = %.4f | E150 = %.4f' % (
                  exact_matchs[0].avg * 100, exact_matchs[2].avg * 100, exact_matchs[4].avg * 100, exact_matchs[9].avg * 100,
                  exact_matchs[19].avg * 100, exact_matchs[49].avg * 100, exact_matchs[99].avg * 100, exact_matchs[149].avg * 100)
              )
        print('Eval:'
              + '\n      E1 = %.4f | E3 = %.4f | E5 = %.4f | E10 = %.4f | E20 = %.4f | E50 = %.4f | E100 = %.4f |' % (
                  exact_matchs_2[0].avg * 100, exact_matchs_2[2].avg * 100, exact_matchs_2[4].avg * 100,
                  exact_matchs_2[9].avg * 100,
                  exact_matchs_2[19].avg * 100, exact_matchs_2[49].avg * 100, exact_matchs_2[99].avg * 100,)
              )


def trim_nqsub_200to50(splt):
    dat50 = []
    with open('/u/pandu/data/openQA/data/datasets/nq/dpr_retrieval_sub200/nq-'+splt+'.json', 'r') as ns200f:
        dat = json.load(ns200f)
        for que in tqdm(dat):
            que['ctxs'] = que['ctxs'][:50]
            dat50.append(que)
    print('Writing file...')
    with open('/u/pandu/data/openQA/data/datasets/nq/dpr_retrieval_sub50/nq-'+splt+'.json', 'w+') as ns50f:
        ns50f.write(json.dumps(dat50, indent=4) + '\n')
    print('Success!')



# dset = 'quasart' # {'p95': 39, 'p96': 39, 'p97': 40, 'p98': 41, 'p99': 43, 'p995': 44, 'p999': 50, 'p100': 125}
# dset = 'unftriviaqa' # {'p95': 64, 'p96': 69, 'p97': 76, 'p98': 89, 'p99': 118, 'p995': 165, 'p999': 635, 'p100': 178876}
# dset = 'nqsub' # {'p95': 132, 'p96': 133, 'p97': 135, 'p98': 139, 'p99': 146, 'p995': 157, 'p999': 196, 'p100': 712}
# dset = 'searchqa' # {'p95': 52, 'p96': 53, 'p97': 54, 'p98': 56, 'p99': 58, 'p995': 62, 'p999': 74, 'p100': 169}
# dset = 'webquestions' # {'p95': 231, 'p96': 245, 'p97': 263, 'p98': 289, 'p99': 337, 'p995': 392, 'p999': 549, 'p100': 1511}
dset = 'trec' # {'p95': 221, 'p96': 234, 'p97': 250, 'p98': 273, 'p99': 316, 'p995': 359, 'p999': 476, 'p100': 1142}
# dset = 'webquestions100'

# splt = 'test'
splt = 'train'
# splt = 'dev'
# splt = 'debug'

dfile = '/u/pandu/data/openQA/data/datasets/'+dset+'/'+splt+'.json'
afile = '/u/pandu/data/openQA/data/datasets/'+dset+'/'+splt+'.txt'
sfile = '/u/pandu/data/openQA/data/datasets/'+dset+'/'+splt+'_utf8.norm_tok_orig.stat'
paragraph_len_stat(dfile, sfile)
# answer_rate(dfile, afile)
# avg_paragraph_len(sfile)
# avg length: quasart 22, unftriviaqa 36, nqsub 118, searchqa 38, webquestions 102, trec 101
# p95 length: quasart 39, unftriviaqa 64, nqsub 132, searchqa 52, webquestions 231, trec 221

# trim_nqsub_200to50('train')