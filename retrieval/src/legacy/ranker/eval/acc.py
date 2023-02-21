from utils import data, vector
import logging
import re
import string
from collections import Counter

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


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------
def validate_unofficial_with_doc(args, data_loader, model, global_stats, exs_with_doc, docs_by_question, mode):
    """Run one full unofficial validation with docs.
    Unofficial = doesn't use SQuAD script.
    """
    eval_time = data.Timer()
    f1 = data.AverageMeter()
    exact_match = data.AverageMeter()

    out_set = set({33, 42, 45, 70, 39})
    logger.info("validate_unofficial_with_doc")
    # Run through examples

    examples = 0
    aa = [0.0 for i in range(args.default_num_docs)]
    bb = [0.0 for i in range(args.default_num_docs)]
    aa_sum = 0.0
    display_num = 10
    for idx, ex_with_doc in enumerate(data_loader):
        ex = ex_with_doc[0]
        batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]

        scores_doc_num, scores_pos, pos = model.predict_top_n(ex_with_doc, top_n=10)

        scores = [{} for i in range(batch_size)]

        for idx_doc in range(0, args.default_num_docs):

            pred_s, pred_e = pos[idx_doc]  # batch_size (4) * span_size (top_n) 2-d list of tuples
            pred_score = scores_pos[:, :, idx_doc]
            for i in range(batch_size):
                doc_text = docs_by_question[ex_id[i]][idx_doc % len(docs_by_question[ex_id[i]])]["document"]
                # has_answer_t = data.has_answer(exs_with_doc[ex_id[i]]['answer'], doc_text)

                for k in range(10):
                    try:
                        prediction = []
                        for j in range(pred_s[i][k], pred_e[i][k] + 1):
                            prediction.append(doc_text[j])
                        prediction = " ".join(prediction).lower()
                        if (prediction not in scores[i]):
                            scores[i][prediction] = 0
                        scores[i][prediction] += pred_score[i][k] * scores_doc_num[i][0][idx_doc]
                    except:
                        pass

        for i in range(batch_size):
            _, indices = scores_doc_num[i].sort(1, descending=True)
            for j in range(0, display_num):
                idx_doc = indices[0][j]
                doc_text = docs_by_question[ex_id[i]][idx_doc % len(docs_by_question[ex_id[i]])]["document"]
                if (data.has_answer(exs_with_doc[ex_id[i]]['answer'], doc_text)[0]):
                    aa[j] = aa[j] + 1
                bb[j] = bb[j] + 1

        for i in range(batch_size):

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
            exact_match.update(metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths))
            f1.update(metric_max_over_ground_truths(
                f1_score, prediction, ground_truths))
            # a = sorted(scores[i].items(), key=lambda d: d[1], reverse=True)

        examples += batch_size
        if (mode == "train" and examples >= 1000):
            break
    try:
        for j in range(0, display_num):
            if (j > 0):
                aa[j] = aa[j] + aa[j - 1]
                bb[j] = bb[j] + bb[j - 1]
            logger.info(aa[j] / bb[j])
    except:
        pass
    logger.info('%s valid official with doc: Epoch = %d | EM = %.2f | ' %
                (mode, global_stats['epoch'], exact_match.avg * 100) +
                'F1 = %.2f | examples = %d | valid time = %.2f (s)' %
                (f1.avg * 100, examples, eval_time.time()))

    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}
