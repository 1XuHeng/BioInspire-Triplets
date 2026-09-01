import json
import unicodedata
from collections import Counter


PREDICTION_FILE = "./outputs/llama/lora_infer/predictions.jsonl"
# PREDICTION_FILE = "./outputs/llama/base_infer/predictions.jsonl"

FIELDS = [
    "source",
    "prototype",
    "application",
]


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def normalize(value):
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    return " ".join(value.strip().split())


def relation_key(rel, kind):
    if kind == "triplet":
        return tuple(
            normalize(rel.get(field))
            for field in FIELDS
        )
    return normalize(rel.get(kind))


def deduplicate_relations(relations):
    unique = []
    seen = set()

    for rel in relations:
        key = relation_key(rel, "triplet")
        if key not in seen:
            seen.add(key)
            unique.append(rel)

    return unique


def calc_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def count_field(gold, pred, field):
    gold_counter = Counter(
        relation_key(rel, field)
        for rel in gold
    )

    pred_counter = Counter(
        relation_key(rel, field)
        for rel in pred
    )

    matched = gold_counter & pred_counter

    tp = sum(matched.values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(gold_counter.values()) - tp

    return tp, fp, fn


def count_triplet(gold, pred):
    gold_set = {
        relation_key(rel, "triplet")
        for rel in gold
    }

    pred_set = {
        relation_key(rel, "triplet")
        for rel in pred
    }

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    return tp, fp, fn


def add_counts(total, current):
    total[0] += current[0]
    total[1] += current[1]
    total[2] += current[2]


def print_result(name, counts):
    tp, fp, fn = counts
    precision, recall, f1 = calc_prf(tp, fp, fn)

    print(
        f"{name:<16}"
        f"{precision:>12.4f}"
        f"{recall:>12.4f}"
        f"{f1:>12.4f}"
        f"{tp:>10}"
        f"{fp:>10}"
        f"{fn:>10}"
    )


def print_table(title, field_counts, triplet_counts):
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)

    print(
        f"{'Level':<16}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'TP':>10}"
        f"{'FP':>10}"
        f"{'FN':>10}"
    )

    print("-" * 82)

    print_result(
        "Source",
        field_counts["source"],
    )

    print_result(
        "Prototype",
        field_counts["prototype"],
    )

    print_result(
        "Application",
        field_counts["application"],
    )

    print_result(
        "Triplet",
        triplet_counts,
    )

    print("=" * 82)


def main():
    predictions = load_jsonl(
        PREDICTION_FILE
    )

    strict_field = {
        field: [0, 0, 0]
        for field in FIELDS
    }

    strict_triplet = [0, 0, 0]

    for item in predictions:
        gold = item.get("gold", [])

        pred = (
            item.get("parsed_prediction")
            or []
        )

        pred = deduplicate_relations(
            pred
        )

        for field in FIELDS:
            add_counts(
                strict_field[field],
                count_field(
                    gold,
                    pred,
                    field,
                ),
            )

        add_counts(
            strict_triplet,
            count_triplet(
                gold,
                pred,
            ),
        )

    print_table(
        "STRICT MATCH",
        strict_field,
        strict_triplet,
    )


if __name__ == "__main__":
    main()