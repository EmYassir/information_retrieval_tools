from utils import data_tok as data
import logging
import re
import string
import numpy as np
from eval import ndcgr
import json
import torch
from tqdm import tqdm
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Evaluation. Follows official evalutation script for v1.1 of the SQuAD dataset.
# ------------------------------------------------------------------------------
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def topk_scope_match_score(predictions, ground_truth, max_len):
    for i, prediction in enumerate(predictions):
        if normalize_answer(ground_truth) in normalize_answer(prediction):
            return [0] * i + [1] * (max_len - i)
    return [0] * max_len


def topk_metric_max_over_ground_truths(metric_fn, predictions, ground_truths, max_len):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    if len(ground_truths) == 0:
        return [0.0] * max_len
    for ground_truth in ground_truths:
        score = metric_fn(predictions, ground_truth, max_len)
        scores_for_ground_truths.append(score)
    scores_for_ground_truths = np.array(scores_for_ground_truths)
    return np.max(scores_for_ground_truths, 0).tolist()


def topk_ndcg_rels(predictions, ground_truths, max_len):
    rels = [0.0] * max_len
    for i, prediction in enumerate(predictions):
        for ground_truth in ground_truths:
            if normalize_answer(ground_truth) in normalize_answer(prediction):
                rels[i] = 1.0
                break
    return rels


def topk_ndcg_max_over_ground_truths(metric_fn, predictions, ground_truths, max_len):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    if len(ground_truths) == 0:
        return [0.0] * max_len
    rels = metric_fn(predictions, ground_truths, max_len)
    ndcgs = [0.0] * len(rels)
    for i in range(len(rels)):
        ndcg_i = ndcgr.NDCG(i+1, 'identity')
        ndcgs[i] = ndcg_i.evaluate(rels[:i+1])
    return ndcgs


# @profile
def topk_eval_unofficial_with_doc(args, data_loader, model, global_stats, exs_with_doc, docs_by_question, mode, topk):
    """Run one full unofficial validation with docs.
    Unofficial = doesn't use SQuAD script.
    """
    max_len = args.default_num_docs
    eval_time = data.Timer()
    exact_matchs = [data.AverageMeter() for i in range(max_len)]
    ndcgs = [data.AverageMeter() for i in range(max_len)]

    logger.info("eval_unofficial_with_doc")
    # Run through examples
    # top_m = 10  # top_m documents for HITS evaluation
    examples = 0

    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]

        batch_size, _, ex_id = ex[0].size(0), ex[3], ex[-1]
        if isinstance(model, torch.nn.DataParallel):
            scores_docs = model.module.predict(ex_with_doc)
        else:
            scores_docs = model.predict(ex_with_doc)
        scores_docs = scores_docs.detach()
        _, indices = scores_docs.sort(2, descending=True)
        for idx_q in range(batch_size):

            predictions = []
            for j in range(len(indices[idx_q, 0, :])):
                idx_doc = indices[idx_q, 0, j]
                doc_text = docs_by_question[ex_id[idx_q]][idx_doc % len(docs_by_question[ex_id[idx_q]])]["document"]
                predictions.append(" ".join(doc_text))

            ground_truths = []
            answers = exs_with_doc[ex_id[idx_q]]['answers']
            if (args.dataset == "CuratedTrec"):
                ground_truths = answers
            else:
                ground_truths = answers
                # for a in answers:
                #     ground_truths.append(" ".join([w for w in a]))

            # exact_match.update(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
            mtsi = topk_metric_max_over_ground_truths(topk_scope_match_score, predictions, ground_truths, max_len)
            ndcgsi = topk_ndcg_max_over_ground_truths(topk_ndcg_rels, predictions, ground_truths, max_len)
            for j in range(max_len):
                exact_matchs[j].update(mtsi[j])
                ndcgs[j].update(ndcgsi[j])

        examples += batch_size

    logger.info('%s official with doc: Epoch = %d |' % (mode, global_stats['epoch'])
                + '\n        E1 = %.2f | E3 = %.2f | E5 = %.2f | E10 = %.2f | E30 = %.2f |' % (exact_matchs[0].avg * 100, exact_matchs[2].avg * 100, exact_matchs[4].avg * 100, exact_matchs[9].avg * 100, exact_matchs[29].avg * 100)
                + '\n        N1 = %.2f | N3 = %.2f | N5 = %.2f | N10 = %.2f | N30 = %.2f |' % (ndcgs[0].avg * 100, ndcgs[2].avg * 100, ndcgs[4].avg * 100, ndcgs[9].avg * 100, ndcgs[29].avg * 100)
                + '\n        examples = %d | valid time = %.1fs' % (examples, eval_time.time()))

    return {'exact_match': exact_matchs[0].avg * 100, 'n5': ndcgs[0]}


def topk_eval_unofficial_with_doc_rank(num_docs, docfile, ansfile):
    """Run one full unofficial validation with docs.
    Unofficial = doesn't use SQuAD script.
    """
    max_len = num_docs
    eval_time = data.Timer()
    exact_matchs = [data.AverageMeter() for i in range(max_len)]
    ndcgs = [data.AverageMeter() for i in range(max_len)]

    answers = []
    with open(ansfile, 'r') as af:
        for line in af:
            axs = json.loads(line)
            answers.append(axs["answers"])

    logger.info("eval_unofficial_with_doc")
    # Run through examples
    # top_m = 10  # top_m documents for HITS evaluation
    examples = 0
    total = sum(1 for line in open(docfile, 'r'))
    with open(docfile, 'r') as df:
        # for idx_q, line in enumerate(tqdm(df, total=total)):
        for idx_q, line in enumerate(df):
            exs = json.loads(line)
            predictions = []

            for idx_doc in range(len(exs)):
                doc_text = exs[idx_doc]["document"]
                predictions.append(" ".join(doc_text))

            ground_truths = []
            answer_good = [exs[0]["answers"]]
            answer = [answers[idx_q]]

            if answer != answer_good:
                print('x', end=' > ')
            print(idx_q, answer, answer_good)
                # raise Exception
            for a in answer:
                ground_truths.append(" ".join(a))

            # exact_match.update(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
            mtsi = topk_metric_max_over_ground_truths(topk_scope_match_score, predictions, ground_truths, max_len)
            ndcgsi = topk_ndcg_max_over_ground_truths(topk_ndcg_rels, predictions, ground_truths, max_len)
            for j in range(max_len):
                exact_matchs[j].update(mtsi[j])
                ndcgs[j].update(ndcgsi[j])

            examples += 1

    logger.info('Evaluation from file: '
                + '\n        E1 = %.2f | E3 = %.2f | E5 = %.2f | E10 = %.2f | E30 = %.2f |' % (exact_matchs[0].avg * 100, exact_matchs[2].avg * 100, exact_matchs[4].avg * 100, exact_matchs[9].avg * 100, exact_matchs[29].avg * 100)
                + '\n        N1 = %.2f | N3 = %.2f | N5 = %.2f | N10 = %.2f | N30 = %.2f |' % (ndcgs[0].avg * 100, ndcgs[2].avg * 100, ndcgs[4].avg * 100, ndcgs[9].avg * 100, ndcgs[29].avg * 100)
                + '\n        examples = %d | valid time = %.1fs' % (examples, eval_time.time()))

    return {'exact_match': exact_matchs[0].avg * 100, 'n5': ndcgs[0]}