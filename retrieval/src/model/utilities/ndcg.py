"""F1 metric."""

from sklearn.metrics import ndcg_score

import datasets
import numpy as np

_DESCRIPTION = """
Compute Normalized Discounted Cumulative Gain.

Sum the true scores ranked in the order induced by the predicted scores, after applying a logarithmic discount. Then divide by the best possible score (Ideal DCG, obtained for a perfect ranking) to obtain a score between 0 and 1.

"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    labels (`list` of `int`): The set of labels to include when `average` is not set to `'binary'`, and the order of the labels if `average` is `None`. Labels present in the data can be excluded, for example to calculate a multiclass average ignoring a majority negative class. Labels not present in the data will result in 0 components in a macro average. For multilabel targets, labels are column indices. By default, all labels in `predictions` and `references` are used in sorted order. Defaults to None.
    pos_label (`int`): The class to be considered the positive class, in the case where `average` is set to `binary`. Defaults to 1.
    average (`string`): This parameter is required for multiclass/multilabel targets. If set to `None`, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data. Defaults to `'binary'`.
        - 'binary': Only report results for the class specified by `pos_label`. This is applicable only if the classes found in `predictions` and `references` are binary.
        - 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
        - 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        - 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters `'macro'` to account for label imbalance. This option can result in an F-score that is not between precision and recall.
        - 'samples': Calculate metrics for each instance, and find their average (only meaningful for multilabel classification).
    sample_weight (`list` of `float`): Sample weights Defaults to None.
Returns:
    NDCG (`float` or `array` of `float`): NDCG score.
"""


_CITATION = """
@article{scikit-learn,
    title={Scikit-learn: Machine Learning in {P}ython},
    author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
           and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
           and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
           Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
    journal={Journal of Machine Learning Research},
    volume={12},
    pages={2825--2830},
    year={2011}
}
"""

HITS_LIST = [1, 3, 5, 20, 50]


class NCDG(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("float32"),
                    "em_labels": datasets.Value("float32"),
                    "ids": datasets.Value("int32"),
                }
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"
            ],
        )

    def _compute(self, predictions, references, em_labels, ids):
        def hits_at_n(y_true, y_pred, k=10):
            scores = []
            arr_pred = np.asarray(y_pred)
            for i, pred in enumerate(arr_pred):
                n_positives = sum(y_true[i])
                if n_positives == 0:
                    continue
                sorted_indexes = np.argsort(pred)[::-1]
                score, size = 0.0, min(k, len(pred))
                for j in range(size):
                    if y_true[i][sorted_indexes[j]] == 1:
                        score += 1.0
                score /= min(size, n_positives)
                scores.append(score)
            return sum(scores) / len(scores)

        agg_preds = {}
        agg_refs = {}
        agg_ems = {}
        for pred, ref, em, id in zip(predictions, references, em_labels, ids):
            if id not in agg_preds:
                agg_preds[id] = []
                agg_refs[id] = []
                agg_ems[id] = []
            agg_preds[id].append(pred)
            agg_refs[id].append(ref)
            agg_ems[id].append(em)

        agg_preds = [
            agg_preds[key] if key in agg_preds.keys() else []
            for key in range(len(agg_preds.keys()))
        ]
        agg_refs = [
            agg_refs[key] if key in agg_refs.keys() else []
            for key in range(len(agg_refs.keys()))
        ]
        agg_ems = [
            agg_ems[key] if key in agg_ems.keys() else []
            for key in range(len(agg_ems.keys()))
        ]

        score_list = [hits_at_n(agg_refs, agg_preds, k=k) for k in HITS_LIST]
        em_score = hits_at_n(agg_ems, agg_preds, k=1)
        ret = {"hits@" + str(hit): score for hit, score in zip(HITS_LIST, score_list)}
        ret.update({"EM_score": em_score})
        return ret
