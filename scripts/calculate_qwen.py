import json
from collections import Counter


PREDICTION_FILE = "./outputs/qwen/lora_infer/predictions.jsonl"
# PREDICTION_FILE = "./outputs/qwen/base_infer/predictions.jsonl"

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
    return str(value).strip()


def relation_tuple(rel):
    return tuple(
        normalize(rel.get(field))
        for field in FIELDS
    )


def calc_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def count_triples(gold, pred):
    gold_counter = Counter(
        relation_tuple(x)
        for x in gold
    )

    pred_counter = Counter(
        relation_tuple(x)
        for x in pred
    )

    matched = gold_counter & pred_counter

    tp = sum(matched.values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(gold_counter.values()) - tp

    return tp, fp, fn


def count_field(gold, pred, field):
    gold_counter = Counter(
        normalize(x.get(field))
        for x in gold
    )

    pred_counter = Counter(
        normalize(x.get(field))
        for x in pred
    )

    matched = gold_counter & pred_counter

    tp = sum(matched.values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(gold_counter.values()) - tp

    return tp, fp, fn


def print_result(name, counts):
    tp, fp, fn = counts
    precision, recall, f1 = calc_prf(tp, fp, fn)

    print(
        f"{name:<20}"
        f"TP={tp:<5} "
        f"FP={fp:<5} "
        f"FN={fn:<5} "
        f"Precision={precision * 100:6.2f}%  "
        f"Recall={recall * 100:6.2f}%  "
        f"F1={f1 * 100:6.2f}%"
    )


predictions = load_jsonl(
    PREDICTION_FILE
)


strict_triple = [0, 0, 0]

strict_field = {
    field: [0, 0, 0]
    for field in FIELDS
}


for item in predictions:
    gold = item.get("gold", [])
    pred = item.get("parsed_prediction") or []

    tp, fp, fn = count_triples(
        gold,
        pred,
    )

    strict_triple[0] += tp
    strict_triple[1] += fp
    strict_triple[2] += fn

    for field in FIELDS:
        tp, fp, fn = count_field(
            gold,
            pred,
            field,
        )

        strict_field[field][0] += tp
        strict_field[field][1] += fp
        strict_field[field][2] += fn


print("=" * 100)
print("STRICT MATCH")
print("=" * 100)

for field in FIELDS:
    print_result(
        field,
        strict_field[field],
    )

print_result(
    "triplet",
    strict_triple,
)