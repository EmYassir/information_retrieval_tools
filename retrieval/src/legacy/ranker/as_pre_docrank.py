'''
Baseline document ranking method bm25, orig, using method 'baseline_rank_data'
Evaluate ranked documents 'eval_ranked_doc'
Execute before pre_txttoken.py
'''
import io
import json
from rank_bm25 import BM25Okapi
import numpy as np
from utils.data import *
from tqdm import tqdm
from utils import data
import logging

logger = logging.getLogger(__name__)


def topk_hits_max_over_ground_truths(hit_order_over_ground, max_len):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    for i, hit in enumerate(hit_order_over_ground):
        if hit:
            return [0] * i + [1] * (max_len - i)
    return [0] * max_len


def eval_ranked_doc(docfile, ansfile, max_len=100):
    """Run one full unofficial validation with docs.
    Unofficial = doesn't use SQuAD script.
    """
    exact_matchs = [data.AverageMeter() for i in range(max_len)]

    answers = []
    with open(ansfile, 'r') as af:
        for line in af:
            if "\\" in line:
                nline = line.replace("\\", "\\\\")
            else:
                nline = line
            axs = json.loads(nline.strip())
            # axs = json.loads(line)
            answers.append(axs["answers"])

    logger.info("eval_unofficial_with_doc")
    # Run through examples
    # top_m = 10  # top_m documents for HITS evaluation
    examples = 0
    total = sum(1 for line in open(docfile, 'r'))
    with open(docfile, 'r') as df:
        for idx_q, line in enumerate(tqdm(df, total=total)):
            # for idx_q, line in enumerate(df):
            exs = json.loads(line)
            hits = []

            for idx_doc in range(len(exs)):
                rel = 1 if exs[idx_doc]["has_answers"][0] else 0
                hits.append(rel)

            mtsi = topk_hits_max_over_ground_truths(hits, max_len)
            for j in range(max_len):
                exact_matchs[j].update(mtsi[j])

            examples += 1

        print('Eval:'
              + '\n      E1 = %.4f | E3 = %.4f | E5 = %.4f | E10 = %.4f | E20 = %.4f | E50 = %.4f | E100 = %.4f |' % (
                  exact_matchs[0].avg * 100, exact_matchs[2].avg * 100, exact_matchs[4].avg * 100,
                  exact_matchs[9].avg * 100,
                  exact_matchs[19].avg * 100, exact_matchs[49].avg * 100, exact_matchs[99].avg * 100,)
              + '\n      examples = %d ' % total)

        return {'exact_match': exact_matchs[0].avg * 100}


def baseline_rank_data(docfile, afile, doc_bmfile, mtype='orig', top_n=20):
    answers = []
    with open(afile, 'r') as af:
        for line in af:
            # axs = json.loads(line)
            if "\\" in line:
                nline = line.replace("\\", "\\\\")
            else:
                nline = line
            axs = json.loads(nline.strip())
            answers.append(axs["answers"])
    newlines = []
    cnt = 0
    ant = 0
    total = sum(1 for line in open(docfile, 'r'))
    with io.open(doc_bmfile, 'w+', encoding='utf-8') as rcf:
        with open(docfile, 'r') as df:
            print('Processing file', docfile)
            short_lines = []

            for idx_line, line in enumerate(tqdm(df, total=total)):
                # if len(newlines) % 1000 == 0 or idx_line == total:
                #     rcf.writelines(newlines)
                #     newlines = []

                exs = json.loads(line)
                corpus = [ex['document'] for ex in exs]
                ids = [ex['id'] for ex in exs]
                answers_in_corpus =  [ex['has_answers'] for ex in exs]
                scores = [0.0] * len(exs)
                for l, _ in answers_in_corpus:
                    if l:
                        cnt += 1
                        break
                query = exs[0]['question']
                # corpus_id = exs[0]['id'][0]
                len_inds = min(top_n, len(corpus))
                if mtype == 'bm25':
                    # pass
                    bm25o = BM25Okapi(corpus)
                    scores = bm25o.get_scores(query)
                    top_n_inds = np.argsort(scores)[::-1][:len_inds]
                else:
                    top_n_inds = [i for i in range(len_inds)]
                if top_n > len(corpus):
                    short_lines.append(idx_line)
                for t in top_n_inds:
                    if answers_in_corpus[t][0]:
                        ant += 1
                        break
                ranked_line = [
                    {"question": query, "document": corpus[i], "id": ids[i], "has_answers": answers_in_corpus[i],
                     "answers": answers[idx_line], "score": scores[i]} for i in top_n_inds]
                # newlines.append(json.dumps(ranked_line)+'\n')
                rcf.write(json.dumps(ranked_line) + '\n')
            print('total', total, 'orig_has_answer', cnt, 'after_has_answer', ant)
            print('short lines in total', len(short_lines), ': ', short_lines)

        # print("Writing to file", doc_bmfile)
        # for ln in tqdm(newlines):
        #     rcf.write((json.dumps(ln) + '\n'))
        rcf.flush()
        print('Done!')


dset = 'trecqa_as_abl'

splt = 'train'
# splt = 'dev'
# splt = 'test'

# mtp = 'orig'
mtp = 'bm25'
dfile = '/u/pandu/data/openQA/data/datasets/' + dset + '/' + splt + '_utf8.norm.json'
afile = '/u/pandu/data/openQA/data/datasets/' + dset + '/' + splt + '_utf8.norm.txt'
odfile = '/u/pandu/data/openQA/data/datasets/' + dset + '/' + splt + '_utf8.norm_' + mtp + '.json'
rdfile = '/u/pandu/data/openQA/data/datasets/' + dset + '/' + splt + '_utf8.norm_rank.json'


baseline_rank_data(dfile, afile, odfile, mtype=mtp, top_n=100)

# eval_ranked_doc(dfile, afile, dataset="quasart")
eval_ranked_doc(odfile, afile, dataset=dset, max_len=100)
# eval_ranked_doc(odfile, afile, dataset="webquestions100")
# eval_ranked_doc(rdfile, afile, dataset='quasart')

'''trecqa_AS_ablation'''
# orig_test 100
# E1 = 0.0000 | E3 = 36.7647 | E5 = 69.1176 | E10 = 95.5882 | E20 = 100.0000 | E50 = 100.0000 | E100 = 100.0000 |
# orig_dev 100
# E1 = 0.0000 | E3 = 50.7246 | E5 = 72.4638 | E10 = 97.1014 | E20 = 100.0000 | E50 = 100.0000 | E100 = 100.0000 |
# orig_train 100
# E1 = 0.0000 | E3 = 55.0562 | E5 = 71.9101 | E10 = 84.2697 | E20 = 94.3820 | E50 = 98.8764 | E100 = 100.0000 |

# bm25_test 100
# E1 = 54.4118 | E3 = 85.2941 | E5 = 97.0588 | E10 = 100.0000 | E20 = 100.0000 | E50 = 100.0000 | E100 = 100.0000 |
# bm25_dev 100
# E1 = 46.3768 | E3 = 91.3043 | E5 = 98.5507 | E10 = 100.0000 | E20 = 100.0000 | E50 = 100.0000 | E100 = 100.0000 |
# bm25_train 100
# E1 = 44.9438 | E3 = 89.8876 | E5 = 98.8764 | E10 = 100.0000 | E20 = 100.0000 | E50 = 100.0000 | E100 = 100.0000 |