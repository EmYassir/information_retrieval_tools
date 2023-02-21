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
from eval import accranker
from utils import data
import logging

logger = logging.getLogger(__name__)


def eval_ranked_doc(docfile, ansfile, dataset='quasart',max_len=100):
    """Run one full unofficial validation with docs.
    Unofficial = doesn't use SQuAD script.
    """
    eval_time = data.Timer()
    exact_matchs = [data.AverageMeter() for i in range(max_len)]
    ndcgs = [data.AverageMeter() for i in range(max_len)]

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
            predictions = []

            for idx_doc in range(len(exs)):
                doc_text = exs[idx_doc]["document"]
                predictions.append(" ".join(doc_text))

            ground_truths = []
            if isinstance(answers[idx_q], list):
                answer = answers[idx_q]
            else:
                answer = [answers[idx_q]]
            if dataset == "quasart":
                if "answers" in exs[0]:
                    answer_good = [exs[0]["answers"]]
                    answer_good_t = [" ".join(a) for a in answer_good]
                    answer_t = [" ".join(a) for a in answer]

                    if answer_t != answer_good_t:
                        print('Warning:', idx_q, answer, answer_good)
                        # raise Exception
            if dataset == "CuratedTrec":
                ground_truths = answer
            else:
                for a in answer:
                    ground_truths.append(a)

            # exact_match.update(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))

            mtsi = accranker.topk_metric_max_over_ground_truths(accranker.topk_scope_match_score, predictions, ground_truths, max_len)
            ndcgsi = accranker.topk_ndcg_max_over_ground_truths(accranker.topk_ndcg_rels, predictions, ground_truths, max_len)
            for j in range(max_len):
                exact_matchs[j].update(mtsi[j])
                ndcgs[j].update(ndcgsi[j])

            examples += 1

    print('Eval:'
          + '\n      E1 = %.4f | E3 = %.4f | E5 = %.4f | E10 = %.4f | E20 = %.4f | E50 = %.4f | E100 = %.4f |' % (
          exact_matchs[0].avg * 100, exact_matchs[2].avg * 100, exact_matchs[4].avg * 100, exact_matchs[9].avg * 100,
          exact_matchs[19].avg * 100, exact_matchs[49].avg * 100, exact_matchs[99].avg * 100,)
          + '\n      N1 = %.4f | N3 = %.4f | N5 = %.4f | N10 = %.4f | N20 = %.4f | N50 = %.4f | N100 = %.4f |' % (
          ndcgs[0].avg * 100, ndcgs[2].avg * 100, ndcgs[4].avg * 100, ndcgs[9].avg * 100, ndcgs[19].avg * 100,
          ndcgs[49].avg * 100, ndcgs[99].avg * 100)
          + '\n      examples = %d ' % total)

    return {'exact_match': exact_matchs[0].avg * 100, 'n5': ndcgs[0]}

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
                scores = [ex['score'] for ex in exs]
                ids = [ex['id'] for ex in exs]
                answers_in_corpus = [has_answer(answers[idx_line], doc) for doc in corpus]
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
                ranked_line = [{"question": query, "document": corpus[i], "id": ids[i], "has_answers": answers_in_corpus[i], "answers":answers[idx_line], "score": scores[i]} for i in top_n_inds]
                # newlines.append(json.dumps(ranked_line)+'\n')
                rcf.write(json.dumps(ranked_line)+'\n')
            print('total', total, 'orig_has_answer', cnt, 'after_has_answer', ant)
            print('short lines in total', len(short_lines), ': ', short_lines)


        # print("Writing to file", doc_bmfile)
        # for ln in tqdm(newlines):
        #     rcf.write((json.dumps(ln) + '\n'))
        rcf.flush()
        print('Done!')

'''quasart'''
# dfile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm.json'
# afile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm.txt'
# udfile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm_bm25.json'
# odfile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm_orig.json'
# rdfile = '/u/pandu/data/openQA/data/datasets/quasart/train_utf8.norm_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm.json'
# afile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm.txt'
# udfile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm_bm25.json'
# odfile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm_orig.json'
# rdfile = '/u/pandu/data/openQA/data/datasets/quasart/dev_utf8.norm_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm.json'
# afile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm.txt'
# udfile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm_bm25.json'
# odfile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm_orig.json'
# rdfile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm.json'
# afile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm.txt'
# udfile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm_bm25.json'
# odfile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm_orig.json'
# rdfile = '/u/pandu/data/openQA/data/datasets/quasart/debug_utf8.norm_rank.json'

# dfile = 'resources/debug.json'
# afile = 'resources/debug.txt'
# udfile = 'resources/debug_bm25.json'
# odfile = 'resources/debug_orig.json'
# rdfile = 'resources/debug_rank.json'

'''searchqa'''
# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm.json'
# afile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm.txt'
# udfile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm_bm25.json'
# odfile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm_orig.json'
# rdfile = '/u/pandu/data/openQA/data/datasets/searchqa/train_utf8.norm_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm.json'
# afile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm.txt'
# udfile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm_bm25.json'
# odfile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm_orig.json'
# rdfile = '/u/pandu/data/openQA/data/datasets/searchqa/dev_utf8.norm_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm.json'
# afile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm.txt'
# udfile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm_bm25.json'
# odfile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm_orig.json'
# rdfile = '/u/pandu/data/openQA/data/datasets/searchqa/test_utf8.norm_rank.json'

# dfile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm.json'
# afile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm.txt'
# udfile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm_bm25.json'
# odfile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm_orig.json'
# rdfile = '/u/pandu/data/openQA/data/datasets/searchqa/debug_utf8.norm_rank.json'

# dfile = 'resources/debug.json'
# afile = 'resources/debug.txt'
# udfile = 'resources/debug_bm25.json'
# odfile = 'resources/debug_orig.json'
# rdfile = 'resources/debug_rank.json'

'''nqsub'''
dset = 'nqsub'
# dset = 'webquestions'
# dset = 'webquestions100'
# dset = 'trec100'
# dset = 'trec'
# dset = 'quasart'
# dset = 'trec_d100_p500'
# dset = 'trec100_full'
# dset = 'webquestions100_full'
# dset = 'webquestions_d100_p500'
# dset = 'unftriviaqa100'
# dset = 'trecqa_as_abl'
# dset = 'wqsub'
# dset = 'nqsub200'
dset = 'nqsub50'

splt = 'train'
# splt = 'dev'
# splt = 'debug'
# splt = 'test'

mtp = 'orig'
# mtp = 'bm25'
dfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm.json'
afile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm.txt'
odfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm_'+mtp+'.json'
rdfile = '/u/pandu/data/openQA/data/datasets/'+ dset +'/'+ splt +'_utf8.norm_rank.json'


if dset.endswith('100') or dset == 'trec100_full' or dset == 'webquestions100_full':
    baseline_rank_data(dfile, afile, odfile, mtype=mtp, top_n=100)
elif dset.endswith('500'):
    baseline_rank_data(dfile, afile, odfile, mtype=mtp, top_n=500)
elif 'wqsub' in dset or 'nqsub200' in dset:
    baseline_rank_data(dfile, afile, odfile, mtype=mtp, top_n=200)
else:
    baseline_rank_data(dfile, afile, odfile, mtype=mtp, top_n=50)

# eval_ranked_doc(dfile, afile, dataset="quasart")
eval_ranked_doc(odfile, afile, dataset=dset, max_len=200)
# eval_ranked_doc(odfile, afile, dataset="webquestions100")
# eval_ranked_doc(rdfile, afile, dataset='quasart')


"""
For Reference: document ranking of Quasar-T
Training Set:
    Original Document Ranking Results: 50
          E1 = 21.7227 | E3 = 39.1224 | E5 = 47.9142 | E10 = 58.6485 | E30 = 71.8551 | E50 = 76.0996 | E100 = 80.2605 |
          N1 = 21.7227 | N3 = 31.7622 | N5 = 35.2759 | N10 = 38.5577 | N30 = 41.8002 | N50 = 42.8950 | N100 = 44.2095 |
          examples = 37012
    BM25 Document Ranking Results: 100
          E1 = 18.2211 | E3 = 34.8211 | E5 = 44.6531 | E10 = 57.1517 | E30 = 71.9604 | E50 = 76.2807 | E100 = 80.2605 |
          N1 = 18.2211 | N3 = 27.7411 | N5 = 31.7113 | N10 = 35.7203 | N30 = 39.7966 | N50 = 41.2366 | N100 = 42.9542 |
          examples = 37012
    AQA 100:
          E1 = 36.5692 | E3 = 53.8258 | E5 = 60.9613 | E10 = 69.0398 | E30 = 77.0426 | E50 = 79.0365 | E100 = 80.2578 |
          N1 = 36.5692 | N3 = 46.5755 | N5 = 49.3046 | N10 = 51.6751 | N30 = 53.5473 | N50 = 53.9959 | N100 = 54.2618 |
          examples = 37012 
Development Set:
    Original Document Ranking Results: 50
          E1 = 22.2667 | E3 = 40.3000 | E5 = 49.3000 | E10 = 58.6667 | E30 = 71.9000 | E50 = 75.3667 | E100 = 78.8333 |
          N1 = 22.2667 | N3 = 32.6944 | N5 = 36.3782 | N10 = 39.2903 | N30 = 42.3302 | N50 = 43.1866 | N100 = 44.2925 |
          examples = 3000
    BM25 Document Ranking Results: 100
          E1 = 18.9000 | E3 = 35.1667 | E5 = 44.6667 | E10 = 56.2000 | E30 = 71.1000 | E50 = 75.1000 | E100 = 78.8333 |
          N1 = 18.9000 | N3 = 28.2565 | N5 = 32.0682 | N10 = 35.6513 | N30 = 39.6317 | N50 = 41.0038 | N100 = 42.5572 |
          examples = 3000
    AQA Document Ranking Results: 50
          E1 = 33.8000 | E3 = 50.7667 | E5 = 59.2333 | E10 = 67.6333 | E30 = 75.6667 | E50 = 77.5000 | E100 = 78.8333 |
          N1 = 33.8000 | N3 = 43.6259 | N5 = 47.0176 | N10 = 49.6548 | N30 = 51.8720 | N50 = 52.4368 | N100 = 52.7694 |
          examples = 3000 
Test Set: 
    Orig: 100
      E1 = 20.4333 | E3 = 37.5333 | E5 = 45.8000 | E10 = 56.0667 | E30 = 70.2667 | E50 = 74.7667 | E100 = 79.1000 |
      N1 = 20.4333 | N3 = 30.3542 | N5 = 33.6463 | N10 = 36.8579 | N30 = 40.3934 | N50 = 41.5938 | N100 = 43.1304 |
      examples = 3000 
    BM25: 100
          E1 = 17.3000 | E3 = 33.7333 | E5 = 44.2000 | E10 = 56.5667 | E30 = 70.4333 | E50 = 70.4333 | E100 = 70.4333 |
          N1 = 17.3000 | N3 = 26.7152 | N5 = 30.9542 | N10 = 35.0300 | N30 = 39.0024 | N50 = 39.0024 | N100 = 39.0024 |
          examples = 3000 
    AQA Dcoument Ranking Results: 100
          E1 = 34.7333 | E3 = 51.7333 | E5 = 58.3333 | E10 = 66.5000 | E30 = 76.0000 | E50 = 77.9000 | E100 = 79.1000 |
          N1 = 34.7333 | N3 = 44.6441 | N5 = 47.1220 | N10 = 49.5659 | N30 = 51.8825 | N50 = 52.3626 | N100 = 52.6739 |
          examples = 3000
Debug Set:
    Orig:
          E1 = 32.0000 | E3 = 48.0000 | E5 = 48.0000 | E10 = 58.0000 | E30 = 78.0000 | E50 = 84.0000 | E100 = 84.0000 |
          N1 = 32.0000 | N3 = 40.7565 | N5 = 40.3710 | N10 = 43.7232 | N30 = 49.4817 | N50 = 50.5346 | N100 = 50.5346 |
          examples = 50
    BM25:
          E1 = 22.0000 | E3 = 42.0000 | E5 = 52.0000 | E10 = 58.0000 | E30 = 78.0000 | E50 = 82.0000 | E100 = 86.0000 |
          N1 = 22.0000 | N3 = 33.2382 | N5 = 37.3204 | N10 = 38.2015 | N30 = 44.6739 | N50 = 45.3341 | N100 = 47.4856 |
          examples = 50
    AQA:
          E1 = 36.0000 | E3 = 66.0000 | E5 = 72.0000 | E10 = 78.0000 | E30 = 84.0000 | E50 = 84.0000 | E100 = 84.0000 |
          N1 = 36.0000 | N3 = 52.3093 | N5 = 55.2213 | N10 = 56.5969 | N30 = 57.9678 | N50 = 58.0438 | N100 = 58.0438 |
          examples = 50 
Local Debug Set:
    Orig:
          E1 = 20.0000 | E3 = 40.0000 | E5 = 40.0000 | E10 = 40.0000 | E30 = 60.0000 | E50 = 60.0000 | E100 = 60.0000 |
          N1 = 20.0000 | N3 = 33.8685 | N5 = 34.2453 | N10 = 32.9216 | N30 = 35.0431 | N50 = 35.2769 | N100 = 35.2769 |
          examples = 5
    BM25:
          E1 = 20.0000 | E3 = 40.0000 | E5 = 40.0000 | E10 = 60.0000 | E30 = 60.0000 | E50 = 60.0000 | E100 = 60.0000 |
          N1 = 20.0000 | N3 = 30.0000 | N5 = 31.4128 | N10 = 32.3464 | N30 = 33.8536 | N50 = 34.4254 | N100 = 35.3547 |
          examples = 5 
    AQA Cocument Ranking Results: 100
          E1 = 60.0000 | E3 = 60.0000 | E5 = 60.0000 | E10 = 60.0000 | E30 = 60.0000 | E50 = 60.0000 | E100 = 60.0000 |
          N1 = 60.0000 | N3 = 60.0000 | N5 = 60.0000 | N10 = 60.0000 | N30 = 59.7930 | N50 = 59.6226 | N100 = 59.3990 |
          examples = 5 
"""

'''
Reference Document ranking on nqsub by model trained on quasart
Training Set
    Orig: 50
        E1 = 18.9597 | E3 = 31.5468 | E5 = 37.8764 | E10 = 46.4695 | E30 = 58.2761 | E50 = 62.8600 | E100 = 62.8600 |
        N1 = 18.9597 | N3 = 26.2029 | N5 = 28.7067 | N10 = 31.2075 | N30 = 33.4981 | N50 = 34.1906 | N100 = 34.1906 |
        examples = 79168 
    AQA: 50
        E1 = 17.1736 | E3 = 30.8281 | E5 = 38.8478 | E10 = 50.7465 | E30 = 66.9589 | 2 layers on tokenized
Dev Set
    Orig: 50
        E1 = 14.0687 | E3 = 24.4833 | E5 = 30.4214 | E10 = 38.1409 | E30 = 50.4625 | E50 = 55.1787 | E100 = 55.1787 |
        N1 = 14.0687 | N3 = 20.1500 | N5 = 22.5285 | N10 = 24.8080 | N30 = 27.3806 | N50 = 28.1924 | N100 = 28.1924 |
        examples = 8757 
    AQA: 50
        E1 = 15.4619 | E3 = 26.2875 | E5 = 32.9793 | E10 = 42.2291 | E30 = 56.5833 | 1 layer on raw
        E1 = 14.5712 | E3 = 25.9335 | E5 = 32.7395 | E10 = 43.2797 | E30 = 58.6045 | 2 layers on tokenized
Test Set
    Orig: 50
        E1 = 8.2825 | E3 = 14.4044 | E5 = 17.8670 | E10 = 22.0776 | E30 = 30.3601 | E50 = 34.3213 | E100 = 34.3213 |
        N1 = 8.2825 | N3 = 11.8216 | N5 = 13.2695 | N10 = 14.5563 | N30 = 16.3052 | N50 = 17.0754 | N100 = 17.0754 |
        examples = 3610 
    AQA: 50
        E1 = 17.8670 | E3 = 30.8310 | E5 = 38.0886 | E10 = 46.7590 | E30 = 59.9446 | 1 layer on raw
        E1 = 15.8449 | E3 = 26.9529 | E5 = 34.6537 | E10 = 44.9584 | E30 = 60.6648 | 2 layers on tokenized
Debug Set
    Orig: 50
        E1 = 18.0000 | E3 = 27.0000 | E5 = 35.0000 | E10 = 41.0000 | E30 = 55.0000 | E50 = 58.0000 | E100 = 58.0000 |
        N1 = 18.0000 | N3 = 23.3987 | N5 = 26.3825 | N10 = 28.2639 | N30 = 30.6116 | N50 = 30.7873 | N100 = 30.7873 |
        examples = 100
    AQA: 50
        E1 = 14.0000 | E3 = 27.0000 | E5 = 32.0000 | E10 = 43.0000 | E30 = 61.0000 | 1 layer on raw
        E1 = 11.0000 | E3 = 26.0000 | E5 = 33.0000 | E10 = 47.0000 | E30 = 59.0000 | 2 layers on tokenized
    
'''

'''
Document ranking on nqsub by model trained on quasart
Training Set
    Orig: 50
        E1 = 18.9597 | E3 = 31.5468 | E5 = 37.8764 | E10 = 46.4695 | E30 = 58.2761 | E50 = 62.8600 | E100 = 62.8600 |
        N1 = 18.9597 | N3 = 26.2029 | N5 = 28.7067 | N10 = 31.2075 | N30 = 33.4981 | N50 = 34.1906 | N100 = 34.1906 |
        examples = 79168 
    AQA: 50
        E1 = 17.1736 | E3 = 30.8281 | E5 = 38.8478 | E10 = 50.7465 | E30 = 66.9589 | 2 layers on tokenized by squart
        E1 = 28.6201 | E3 = 43.0439 | E5 = 50.0581 | E10 = 58.5097 | E30 = 68.1273 | 1 layer on tokenized by nqsub
Dev Set
    Orig: 50
        E1 = 14.0687 | E3 = 24.4833 | E5 = 30.4214 | E10 = 38.1409 | E30 = 50.4625 | E50 = 55.1787 | E100 = 55.1787 |
        N1 = 14.0687 | N3 = 20.1500 | N5 = 22.5285 | N10 = 24.8080 | N30 = 27.3806 | N50 = 28.1924 | N100 = 28.1924 |
        examples = 8757 
    AQA: 50
        E1 = 15.4619 | E3 = 26.2875 | E5 = 32.9793 | E10 = 42.2291 | E30 = 56.5833 | 1 layer on raw by quasart
        E1 = 14.5712 | E3 = 25.9335 | E5 = 32.7395 | E10 = 43.2797 | E30 = 58.6045 | 2 layers on tokenized by quasart
        E1 = 21.8668 | E3 = 33.7444 | E5 = 40.3106 | E10 = 48.6811 | E30 = 59.3240 | 1 layer on tokenized by nqsub
Test Set
    Orig: 50
        E1 = 8.2825 | E3 = 14.4044 | E5 = 17.8670 | E10 = 22.0776 | E30 = 30.3601 | E50 = 34.3213 | E100 = 34.3213 |
        N1 = 8.2825 | N3 = 11.8216 | N5 = 13.2695 | N10 = 14.5563 | N30 = 16.3052 | N50 = 17.0754 | N100 = 17.0754 |
        examples = 3610 
    AQA: 50
        E1 = 17.8670 | E3 = 30.8310 | E5 = 38.0886 | E10 = 46.7590 | E30 = 59.9446 | 1 layer on raw by quasart
        E1 = 15.8449 | E3 = 26.9529 | E5 = 34.6537 | E10 = 44.9584 | E30 = 60.6648 | 2 layers on tokenized by quasart
        E1 = 22.3269 | E3 = 34.9584 | E5 = 40.4155 | E10 = 48.6981 | E30 = 60.7202 | 1 layer on tokenized by nqsub
Debug Set
    Orig: 50
        E1 = 18.0000 | E3 = 27.0000 | E5 = 35.0000 | E10 = 41.0000 | E30 = 55.0000 | E50 = 58.0000 | E100 = 58.0000 |
        N1 = 18.0000 | N3 = 23.3987 | N5 = 26.3825 | N10 = 28.2639 | N30 = 30.6116 | N50 = 30.7873 | N100 = 30.7873 |
        examples = 100
    AQA: 50
        E1 = 14.0000 | E3 = 27.0000 | E5 = 32.0000 | E10 = 43.0000 | E30 = 61.0000 | 1 layer on raw by quasart
        E1 = 11.0000 | E3 = 26.0000 | E5 = 33.0000 | E10 = 47.0000 | E30 = 59.0000 | 2 layers on tokenized by quasart
        E1 = 22.0000 | E3 = 39.0000 | E5 = 43.0000 | E10 = 50.0000 | E30 = 61.0000 | 1 layers on tokenized by nqsub

'''

# DPR 200: nqsub200 test
# E1 = 39.2521 | E3 = 53.6011 | E5 = 58.7535 | E10 = 63.9335 | E20 = 68.9474 | E50 = 73.7119 | E100 = 76.3435 |
# tfidf 50: nqsub test
# E1 = 8.2825 | E3 = 14.4044 | E5 = 17.8670 | E10 = 22.0776 | E20 = 27.0914 | E50 = 34.3213 | E100 = 38.0055 |
#dpr 50: nqsub50 test
# E1 = 39.2521 | E3 = 53.6011 | E5 = 58.7535 | E10 = 63.9335 | E20 = 68.9474 | E50 = 73.7119 |
#dpr 50: nqsub50 dev
# E1 = 39.0431 | E3 = 52.7692 | E5 = 57.9651 | E10 = 63.8118 | E20 = 68.0027 | E50 = 72.6505 |
#dpr 50 nqsub50 train
# E1 = 61.0158 | E3 = 74.3861 | E5 = 77.7637 | E10 = 80.7233 | E20 = 82.7228 | E50 = 84.6946 |
