from utils import data
import logging
import re
import string
from collections import Counter
import heapq
import operator
import numpy as np
from eval import ndcgr

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


def f1_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def scope_match_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(predictions, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def topk_f1_score(predictions, ground_truth, max_len):
    """Compute the geometric mean of precision and recall for answer tokens."""
    max_f1s = []
    max_f1 = 0.0
    for i, prediction in enumerate(predictions):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            # f1 = 0.0
            max_f1s.append(max_f1)
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > max_f1:
            max_f1 = f1
        max_f1s.append(max_f1)
    if len(max_f1s) < max_len:
        max_f1s = max_f1s + [max_f1] * (max_len - len(max_f1s))
    return max_f1s


def topk_scope_match_score(predictions, ground_truth, max_len):
    for i, prediction in enumerate(predictions):
        if normalize_answer(ground_truth) in normalize_answer(prediction):
            return [0] * i + [1] * (max_len - i)
    return [0] * max_len


def topk_exact_match_score(predictions, ground_truth, max_len):
    for i, prediction in enumerate(predictions):
        if normalize_answer(ground_truth) == normalize_answer(prediction):
            return [0] * i + [1] * (max_len - i)
    return [0] * max_len


def topk_metric_max_over_ground_truths(metric_fn, predictions, ground_truths, max_len):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(predictions, ground_truth, max_len)
        scores_for_ground_truths.append(score)
    scores_for_ground_truths = np.array(scores_for_ground_truths)
    return np.max(scores_for_ground_truths, 0).tolist()


def topk_ndcg_rels(predictions, ground_truths, max_len):
    rels = [0.0] * max_len
    for i, prediction in enumerate(predictions):
        for ground_truth in ground_truths:
            if normalize_answer(ground_truth) == normalize_answer(prediction):
                rels[i] = 1.0
                break
    return rels


def topk_ndcg_max_over_ground_truths(metric_fn, predictions, ground_truths, max_len):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    rels = metric_fn(predictions, ground_truths, max_len)
    ndcgs = [0.0] * len(rels)
    for i in range(len(rels)):
        ndcg_i = ndcgr.NDCG(i+1, 'identity')
        ndcgs[i] = ndcg_i.evaluate(rels[:i+1])
    return ndcgs

# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------
# @profile
def eval_unofficial_with_doc(args, data_loader, model, global_stats, exs_with_doc, docs_by_question, mode):
    """Run one full unofficial validation with docs.
    Unofficial = doesn't use SQuAD script.
    """
    eval_time = data.Timer()
    exact_match = data.AverageMeter()
    f1 = data.AverageMeter()

    logger.info("eval_unofficial_with_doc")
    # Run through examples
    top_m = 10  # top_m documents for HITS evaluation
    examples = 0

    for idx, ex_with_doc in enumerate(data_loader):
        num_docs = len(ex_with_doc)
        ex = ex_with_doc[0]

        aa = [0.0 for i in range(num_docs)]
        bb = [0.0 for i in range(num_docs)]

        batch_size, _, ex_id = ex[0].size(0), ex[3], ex[-1]
        scores_docs, scores_se, spans_s, spans_e = model.predict_top_n(ex_with_doc, top_n=10)
        scores_docs = scores_docs.detach()
        scores_se = scores_se.detach()
        spans_s = spans_s.detach()
        spans_e = spans_e.detach()

        # Calculate the scores of the top_n predictions (answers) for each document
        scores = [{} for i in range(batch_size)]
        for idx_q in range(batch_size):
            for idx_doc in range(num_docs):
                doc_text = docs_by_question[ex_id[idx_q]][idx_doc % len(docs_by_question[ex_id[idx_q]])]["document"]
                pred_qd_s = spans_s[idx_q, :, idx_doc]
                pred_qd_e = spans_e[idx_q, :, idx_doc]
                pred_qd_score = scores_se[idx_q, :, idx_doc]
                for i in range(len(pred_qd_s)):  # top_n
                    if pred_qd_s[i] < 0 or pred_qd_e[i] < 0:
                        continue
                    try:
                        pred = []
                        for j in range(pred_qd_s[i], pred_qd_e[i] + 1):
                            pred.append(doc_text[j])
                        prediction = " ".join(pred).lower()
                        if prediction not in scores[idx_q]:
                            scores[idx_q][prediction] = 0
                        scores[idx_q][prediction] += pred_qd_score[i] * scores_docs[idx_q, 0, idx_doc]
                    except Exception as excpt:
                        print('err', excpt)
                        pass

        # Calculate the HITS@10 ratio of top-ranked documents of this batch
        for i in range(batch_size):
            _, indices = scores_docs[i].sort(0, descending=True)
            for j in range(0, top_m):
                idx_doc = indices[0, j]
                is_answer = docs_by_question[ex_id[i]][idx_doc % len(docs_by_question[ex_id[i]])]["has_answers"]
                # doc_text = docs_by_question[ex_id[i]][idx_doc%len(docs_by_question[ex_id[i]])]["document"]
                # if data.has_answer(exs_with_doc[ex_id[i]]["answer"], doc_text)[0]:
                if is_answer:
                    aa[j] = aa[j] + 1
                bb[j] = bb[j] + 1

        # Calculate the best prediction (answer) for each question in this batch
        for i in range(batch_size):
            # Best prediction for each question
            best_score = 0
            prediction = ""
            for key in scores[i]:
                if (scores[i][key] > best_score):
                    best_score = scores[i][key]
                    prediction = key
            # Compute metrics
            ground_truths = []
            answer = exs_with_doc[ex_id[i]]['answer']
            if (args.dataset == "CuratedTrec"):
                ground_truths = answer
            else:
                for a in answer:
                    ground_truths.append(" ".join([w for w in a]))
            # logger.info(prediction)
            # logger.info(ground_truths)

            # exact_match.update(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
            exact_match.update(metric_max_over_ground_truths(scope_match_score, prediction, ground_truths))
            f1.update(metric_max_over_ground_truths(f1_score, prediction, ground_truths))

        examples += batch_size
        if (mode == "train" and examples >= 1000):
            break

    # HITS@10
    # try:
    #     for j in range(0, top_m):
    #         if (j > 0):
    #             aa[j] = aa[j] + aa[j - 1]
    #             bb[j] = bb[j] + bb[j - 1]
    #         logger.info('HITS@%d: %.4f', j, aa[j] / bb[j])
    # except:
    #     pass

    logger.info('%s official with doc: Epoch = %d | EM = %.4f | ' %
                (mode, global_stats['epoch'], exact_match.avg * 100) +
                'F1 = %.4f | examples = %d | valid time = %.2f (s)' %
                (f1.avg * 100, examples, eval_time.time()))

    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}


# @profile
def topk_eval_unofficial_with_doc(args, data_loader, model, global_stats, exs_with_doc, docs_by_question, mode, topk):
    """Run one full unofficial validation with docs.
    Unofficial = doesn't use SQuAD script.
    """
    max_len = 30 * 10
    eval_time = data.Timer()
    exact_matchs = [data.AverageMeter() for i in range(max_len)]
    f1s = [data.AverageMeter() for i in range(max_len)]
    ndcgs = [data.AverageMeter() for i in range(max_len)]

    logger.info("eval_unofficial_with_doc")
    # Run through examples
    # top_m = 10  # top_m documents for HITS evaluation
    examples = 0

    for idx, ex_with_doc in enumerate(data_loader):
        num_docs = len(ex_with_doc)
        ex = ex_with_doc[0]

        # aa = [0.0 for i in range(num_docs)]
        # bb = [0.0 for i in range(num_docs)]

        batch_size, _, ex_id = ex[0].size(0), ex[3], ex[-1]
        scores_docs, scores_se, spans_s, spans_e = model.predict_top_n(ex_with_doc, top_n=10)
        # scores_docs = scores_docs.detach()
        # scores_se = scores_se.detach()
        spans_s = spans_s.detach()
        spans_e = spans_e.detach()
        # scores_se_overall =  (scores_se * scores_docs).detach()
        scores_se_overall = scores_se.detach()
        # Calculate the scores of the top_n predictions (answers) for each document
        scores = [{} for i in range(batch_size)]
        for idx_q in range(batch_size):
            for idx_doc in range(num_docs):
                doc_text = docs_by_question[ex_id[idx_q]][idx_doc % len(docs_by_question[ex_id[idx_q]])]["document"]
                pred_qd_s = spans_s[idx_q, :, idx_doc]
                pred_qd_e = spans_e[idx_q, :, idx_doc]
                pred_qd_score = scores_se_overall[idx_q, :, idx_doc]
                for i in range(len(pred_qd_s)):  # top_n
                    if pred_qd_s[i] < 0 or pred_qd_e[i] < 0:
                        continue
                    try:
                        pred = []
                        for j in range(pred_qd_s[i], pred_qd_e[i] + 1):
                            pred.append(doc_text[j])
                        prediction = " ".join(pred).lower()
                        if prediction not in scores[idx_q]:
                            scores[idx_q][prediction] = 0
                        scores[idx_q][prediction] += pred_qd_score[i]
                    except Exception as excpt:
                        print('err', excpt)
                        pass

        # Calculate the HITS@10 ratio of top-ranked documents of this batch
        # for i in range(batch_size):
        #     _, indices = scores_docs[i].sort(0, descending=True)
        #     for j in range(0, top_m):
        #         idx_doc = indices[0, j]
        #         is_answer = docs_by_question[ex_id[i]][idx_doc%len(docs_by_question[ex_id[i]])]["has_answers"]
        #         # doc_text = docs_by_question[ex_id[i]][idx_doc%len(docs_by_question[ex_id[i]])]["document"]
        #         # if data.has_answer(exs_with_doc[ex_id[i]]["answer"], doc_text)[0]:
        #         if is_answer:
        #             aa[j] = aa[j] + 1
        #         bb[j] = bb[j] + 1

        # Calculate the best prediction (answer) for each question in this batch
        for i in range(batch_size):
            # top n prediction for each question
            # sorted_scores_i = sorted(scores[i].items(), key=lambda item: item[1], reverse=True)
            # topk_predictions = [k for k, v in sorted_scores_i[:topk]]
            sorted_scores_i = heapq.nlargest(topk, scores[i].items(), operator.itemgetter(1, 0))
            topk_predictions = [k for k, v in sorted_scores_i]
            # Compute metrics
            ground_truths = []
            answer = exs_with_doc[ex_id[i]]['answer']
            if (args.dataset == "CuratedTrec"):
                ground_truths = answer
            else:
                for a in answer:
                    ground_truths.append(" ".join([w for w in a]))


            # logger.info(prediction)
            # logger.info(ground_truths)
            # exact_match.update(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
            mtsi = topk_metric_max_over_ground_truths(topk_exact_match_score, topk_predictions, ground_truths, max_len)
            f1si = topk_metric_max_over_ground_truths(topk_f1_score, topk_predictions, ground_truths, max_len)
            ndcgsi = topk_ndcg_max_over_ground_truths(topk_ndcg_rels, topk_predictions, ground_truths, max_len)
            for j in range(max_len):
                exact_matchs[j].update(mtsi[j])
                f1s[j].update(f1si[j])

                ndcgs[j].update(ndcgsi[j])

        examples += batch_size
        # if (mode == "train" and examples >= 1000):
        #     break

    # HITS@10
    # try:
    #     for j in range(0, top_m):
    #         if (j > 0):
    #             aa[j] = aa[j] + aa[j - 1]
    #             bb[j] = bb[j] + bb[j - 1]
    #         logger.info('HITS@%d: %.4f', j, aa[j] / bb[j])
    # except:
    #     pass

    logger.info('%s official with doc: Epoch = %d |' % (mode, global_stats['epoch'])
                + '\n                          E1 = %.4f | E3 = %.4f | E5 = %.4f | E10 = %.4f | E30 = %.4f | E150 = %.4f |' % (exact_matchs[0].avg * 100, exact_matchs[2].avg * 100, exact_matchs[4].avg * 100, exact_matchs[9].avg * 100, exact_matchs[29].avg * 100, exact_matchs[149].avg * 100)
                + '\n                          N1 = %.4f | N3 = %.4f | N5 = %.4f | N10 = %.4f | N30 = %.4f | N150 = %.4f |' % (ndcgs[0].avg * 100, ndcgs[2].avg * 100, ndcgs[4].avg * 100, ndcgs[9].avg * 100, ndcgs[29].avg * 100, ndcgs[149].avg * 100)
                + '\n                          F1 = %.4f | F3 = %.4f | F5 = %.4f | F10 = %.4f | F30 = %.4f | F150 = %.4f |' % (f1s[0].avg * 100, f1s[2].avg * 100, f1s[4].avg * 100, f1s[9].avg * 100, f1s[29].avg * 100, f1s[149].avg * 100)
                + '\n                          examples = %d | valid time = %.2f (s)' % (examples, eval_time.time()))

    return {'exact_match': exact_matchs[4].avg * 100, 'f5': f1s[4].avg * 100, 'n5': ndcgs[4]}
