import argparse
import string
from typing import List, Dict, Union

import pandas as pd

from file_io import read_lm_kbc_jsonl


def clean_object(obj: str) -> Union[str, None]:
    """
    Cleans the object by removing punctuation and lower-casing.
    """

    if not obj:
        return None

    for punctuation in string.punctuation:
        obj = obj.replace(punctuation, "")

    return obj.lower().strip()


def is_none_gts(gts: List[List[str]]) -> bool:
    """
    Checks if the ground truth object is none.
    """
    return not gts


def is_none_preds(preds: List[str]) -> bool:
    """
    Checks if the prediction object is none (with relaxing rules).
    """
    return preds is None or len(preds) == 0 or (
            len(preds) == 1 and
            (
                    list(preds)[0] is None or
                    list(preds)[0].lower() in {"", "none", "null"}
            )
    )


def true_positives(preds: List[str], gts: List[List[str]]) -> int:
    """
    Calculates the number of true positives
    for a given pair of subject and relation.
    Method:
        Iterate over the ground truth objects, each is a list of possible
        aliases. For each ground truth object, check if the prediction
        contains any of its aliases. If so, increment the true positives by 1.

    Args:
        preds: list of normalized predictions
        gts: list of ground truth objects (lists of normalized aliases)

    Returns:
        true_positives: int
    """

    tp = 0
    for gt in gts:
        gt_set = set(gt)
        if any(pred in gt_set for pred in preds):
            tp += 1

    return tp


def precision(preds: List[str], gts: List[List[str]]) -> float:
    """
    Calculates the precision of the predictions
    for a given pair of subject and relation.

    Args:
        preds: list of predictions
        gts: list of ground truth objects

    Returns:
        precision: float
    """

    # when nothing is predicted, precision 1 irrespective of the ground truth value
    if is_none_preds(preds):
        return 1

    # When the ground truth object is none
    if is_none_gts(gts):
        return 1.0 if is_none_preds(preds) else 0.0

    # When the ground truth object is not none
    try:
        return min(true_positives(preds, gts) / len(preds), 1.0)
    except TypeError:
        return 0.0


def recall(preds: List[str], gts: List[List[str]]) -> float:
    """
    Calculates the recall of the predictions
    for a given pair of subject and relation.

    Args:
        preds: list of predictions
        gts: list of ground truth objects

    Returns:
        recall: float
    """

    # When the ground truth object is none return 1 even if there are predictions (edge case)
    if is_none_gts(gts):
        return 1.0

    # When the ground truth object is not none
    try:
        return true_positives(preds, gts) / len(gts)
    except TypeError:
        return 0.0


def f_beta_score(p: float, r: float, beta: float = 1) -> float:
    """
    Calculates the F1-score of the predictions
    for a given pair of subject and relation.

    Args:
        p: precision
        r: recall

    Returns:
        f1_score: float
    """
    assert beta != 0
    try:
        return ((1 + beta*beta) * p * r) / (beta*beta*p + r)
    except ZeroDivisionError:
        return 0.0


def rows_to_dict(rows: List[Dict]) -> Dict:
    """
    Index the ground truth/prediction rows by subject entity and relation.
    """

    return {(r["SubjectEntity"], r["Relation"]): r["ObjectEntities"] for r in
            rows}


def evaluate_per_sr_pair(pred_rows, gt_rows, beta=1.0) \
        -> List[Dict[str, float]]:

    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)

    results = []

    for subj, rel in gt_dict:
        # get and normalize the ground truth objects
        gts = []
        for gt in gt_dict[(subj, rel)]:
            gts.append([clean_object(obj) for obj in gt])

        # get and normalize the predictions
        preds = list(set(
            clean_object(obj) for obj in pred_dict.get((subj, rel), [])))

        # calculate the scores
        p = precision(preds, gts)
        r = recall(preds, gts)
        f1 = f_beta_score(p, r, beta=beta)

        results.append({
            "SubjectEntity": subj,
            "Relation": rel,
            "p": p,
            "r": r,
            "f1": f1
        })

        # if p > 1.0 or r > 1.0:
        #     print(f"{subj} {rel} {p} {r} {f1} {gts} {preds}")

    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def combine_scores_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = []
        scores[r["Relation"]].append({
            "p": r["p"],
            "r": r["r"],
            "f1": r["f1"],
        })

    for rel in scores:
        scores[rel] = {
            "p": sum([x["p"] for x in scores[rel]]) / len(scores[rel]),
            "r": sum([x["r"] for x in scores[rel]]) / len(scores[rel]),
            "f1": sum([x["f1"] for x in scores[rel]]) / len(scores[rel]),
        }

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Precision, Recall and F1-score of predictions"
    )

    parser.add_argument(
        "-p",
        "--predictions",
        type=str,
        required=True,
        help="Path to the predictions file (required)"
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        required=True,
        help="Path to the ground truth file (required)"
    )
    parser.add_argument('--beta', type=float, default=1.0)

    args = parser.parse_args()

    pred_rows = read_lm_kbc_jsonl(args.predictions)
    gt_rows = read_lm_kbc_jsonl(args.ground_truth)

    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows, beta=args.beta)
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    scores_per_relation["***Average***"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()]) / len(
            scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()]) / len(
            scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()]) / len(
            scores_per_relation),
    }

    print(pd.DataFrame(scores_per_relation).transpose().round(3))


if __name__ == "__main__":
    main()
