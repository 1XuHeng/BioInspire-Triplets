import os
import json
import re
import traceback
import unicodedata
from collections import Counter

import torch
import torch.distributed as dist

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_PATH = "./model/llama"


# ADAPTER_PATH = "./outputs/llama/lora/final_adapter"
ADAPTER_PATH = None

TEST_FILE = "./data/test.json"

# OUTPUT_DIR = "./outputs/llama/lora_infer"
OUTPUT_DIR = "./outputs/llama/base_infer"

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "predictions.jsonl",
)



MAX_LENGTH = 2048

MAX_NEW_TOKENS = 384


INFERENCE_BATCH_SIZE = 1

SEED = 42

SYSTEM_PROMPT = """You are an information extraction model for bio-inspired design literature.

Extract ALL explicitly stated bio-inspired Source–Prototype–Application relations from the supplied text.

SPAN RULES
- source, prototype, and application must each be a contiguous verbatim span copied from the input.
- Use the MINIMAL span that still identifies the annotated concept/entity.
- Do not paraphrase, normalize, expand abbreviations, change singular/plural forms, or add surrounding context.
- Do not append the biological source to a prototype when the shorter prototype phrase is sufficient.
- Do not append generic engineering context to an application when a shorter application span is sufficient.

RELATION RULES
- Extract every distinct explicit relation.
- If one source/application is linked to multiple distinct prototypes, output one relation per prototype.
- Do not merge distinct prototypes into a single relation.
- Do not invent implicit relations or use external knowledge.

Return ONLY a valid JSON array. Every relation must contain exactly these three fields:
source, prototype, application.
Do not output source_type, prototype_type, application_type, explanations, markdown, or any additional fields.
"""

REQUIRED_OUTPUT_FIELDS = [
    "source",
    "prototype",
    "application",
]

def get_process_rank():
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def setup_distributed():

    if get_world_size() <= 1:
        return False

    local_rank = get_local_rank()

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return True


def cleanup_distributed():

    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        dist.barrier()
        dist.destroy_process_group()

def set_seed(seed):

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_json_file(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{path}: top level must be a JSON array"
        )

    return data


def validate_dataset(
    data,
    path,
):

    for i, item in enumerate(
        data,
        start=1,
    ):

        if "input" not in item:
            raise ValueError(
                f"{path}: item {i} missing input"
            )

        if "output" not in item:
            raise ValueError(
                f"{path}: item {i} missing output"
            )

        if not isinstance(
            item["input"],
            str,
        ):
            raise ValueError(
                f"{path}: item {i} input must be a string"
            )

        if not isinstance(
            item["output"],
            list,
        ):
            raise ValueError(
                f"{path}: item {i} output must be a list"
            )

        input_text = item["input"]

        for j, relation in enumerate(
            item["output"],
            start=1,
        ):

            if not isinstance(
                relation,
                dict,
            ):
                raise ValueError(
                    f"{path}: item {i}, "
                    f"relation {j} must be an object"
                )

            for field in REQUIRED_OUTPUT_FIELDS:

                if field not in relation:

                    raise ValueError(
                        f"{path}: item {i}, "
                        f"relation {j}, "
                        f"missing {field}"
                    )

                if not isinstance(
                    relation[field],
                    str,
                ):

                    raise ValueError(
                        f"{path}: item {i}, "
                        f"relation {j}, "
                        f"{field} must be a string"
                    )

                if relation[field] not in input_text:

                    raise ValueError(
                        f"{path}: item {i}, relation {j}, "
                        f"{field} is not a contiguous "
                        f"verbatim span of input: "
                        f"{relation[field]!r}"
                    )

def _relation_tuple(relation):

    return tuple(
        relation[field]
        for field in REQUIRED_OUTPUT_FIELDS
    )


def canonicalize_output(
    input_text,
    relations,
):

    unique = []

    seen = set()

    for relation in relations:

        normalized = {
            field: relation[field]
            for field in REQUIRED_OUTPUT_FIELDS
        }

        key = _relation_tuple(
            normalized
        )

        if key not in seen:

            seen.add(key)

            unique.append(
                normalized
            )


    def pos(span):

        p = input_text.find(span)

        return (
            p
            if p >= 0
            else 10**12
        )


    unique.sort(
        key=lambda r: (
            min(
                pos(r["source"]),
                pos(r["prototype"]),
                pos(r["application"]),
            ),
            pos(r["source"]),
            pos(r["prototype"]),
            pos(r["application"]),
            r["source"],
            r["prototype"],
            r["application"],
        )
    )

    return unique

def render_chat_template(
    tokenizer,
    messages,
    add_generation_prompt,
):

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

def validate_prediction_schema(obj):

    if not isinstance(obj, list):
        return False

    for rel in obj:

        if not isinstance(
            rel,
            dict,
        ):
            return False

        if set(rel.keys()) != set(
            REQUIRED_OUTPUT_FIELDS
        ):
            return False

        if not all(
            isinstance(
                rel[k],
                str,
            )
            for k in REQUIRED_OUTPUT_FIELDS
        ):
            return False

    return True


def parse_prediction(text):

    try:

        obj = json.loads(
            text.strip()
        )

    except Exception:

        return [], False

    if not validate_prediction_schema(
        obj
    ):
        return [], False

    return obj, True

def normalize_text(value):

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    return " ".join(
        value.strip().split()
    )

def _relation_key(
    rel,
    kind,
):

    s = normalize_text(
        rel["source"]
    )

    p = normalize_text(
        rel["prototype"]
    )

    a = normalize_text(
        rel["application"]
    )


    if kind == "source":
        return s

    if kind == "prototype":
        return p

    if kind == "application":
        return a

    if kind == "triplet":
        return (
            s,
            p,
            a,
        )

    raise ValueError(kind)

def load_model(
    model_path,
    adapter_path,
    device,
):

    rank = get_process_rank()

    tokenizer_source = (
        adapter_path
        if adapter_path
        else model_path
    )


    print(
        f"[Rank {rank}] Loading tokenizer...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"


    if torch.cuda.is_available():

        use_bf16 = (
            torch.cuda.is_bf16_supported()
        )

        model_dtype = (
            torch.bfloat16
            if use_bf16
            else torch.float16
        )

    else:

        model_dtype = torch.float32


    print(
        f"[Rank {rank}] "
        f"Model dtype: {model_dtype}",
        flush=True,
    )

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
    )

    if adapter_path:

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )

    else:

        model = base_model


    model.config.use_cache = True

    model = model.to(device)

    model.eval()

    print(
        f"[Rank {rank}] "
        f"Model loaded.",
        flush=True,
    )

    return model, tokenizer

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    device,
    input_texts,
):

    prompts = []

    for input_text in input_texts:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": input_text,
            },
        ]

        prompts.append(
            render_chat_template(
                tokenizer,
                messages,
                add_generation_prompt=True,
            )
        )


    batch = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    batch = {
        k: v.to(device)
        for k, v in batch.items()
    }

    prompt_width = (
        batch["input_ids"].shape[1]
    )

    eos_token_id = getattr(
        model.generation_config,
        "eos_token_id",
        None,
    )

    if eos_token_id is None:
        eos_token_id = getattr(
            model.config,
            "eos_token_id",
            None,
        )

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    generated = model.generate(
        **batch,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
    )


    continuations = generated[
        :,
        prompt_width:,
    ]


    decoded = tokenizer.batch_decode(
        continuations,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return [
        x.strip()
        for x in decoded
    ]

def _safe_f1(
    tp,
    fp,
    fn,
):

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return (
        precision,
        recall,
        f1,
    )


def compute_extraction_metrics(
    raw_examples,
    prediction_texts,
):

    kinds = [
        "source",
        "prototype",
        "application",
        "triplet",
    ]

    counts = {
        k: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }
        for k in kinds
    }

    valid_json_count = 0

    for example, pred_text in zip(
        raw_examples,
        prediction_texts,
    ):

        gold = canonicalize_output(
            example["input"],
            example["output"],
        )

        pred, valid = parse_prediction(
            pred_text
        )

        valid_json_count += int(valid)

        if not valid:
            pred = []

        # Prediction 先按完整三元组去重，
        # 避免模型重复输出同一 relation 被重复计数。
        unique_pred = []
        seen = set()

        for rel in pred:
            key = _relation_key(
                rel,
                "triplet",
            )

            if key not in seen:
                seen.add(key)
                unique_pred.append(rel)

        pred = unique_pred

        for kind in [
            "source",
            "prototype",
            "application",
        ]:

            gold_counter = Counter(
                _relation_key(
                    rel,
                    kind,
                )
                for rel in gold
            )

            pred_counter = Counter(
                _relation_key(
                    rel,
                    kind,
                )
                for rel in pred
            )

            matched = (
                gold_counter
                & pred_counter
            )

            tp = sum(
                matched.values()
            )

            fp = (
                sum(pred_counter.values())
                - tp
            )

            fn = (
                sum(gold_counter.values())
                - tp
            )

            counts[kind]["TP"] += tp
            counts[kind]["FP"] += fp
            counts[kind]["FN"] += fn

        gold_set = {
            _relation_key(
                rel,
                "triplet",
            )
            for rel in gold
        }

        pred_set = {
            _relation_key(
                rel,
                "triplet",
            )
            for rel in pred
        }

        counts["triplet"]["TP"] += len(
            gold_set & pred_set
        )

        counts["triplet"]["FP"] += len(
            pred_set - gold_set
        )

        counts["triplet"]["FN"] += len(
            gold_set - pred_set
        )

    metrics = {
        "valid_json": (
            valid_json_count
            / max(
                1,
                len(raw_examples),
            )
        ),
    }

    for kind in kinds:

        tp = counts[kind]["TP"]
        fp = counts[kind]["FP"]
        fn = counts[kind]["FN"]

        precision, recall, f1 = _safe_f1(
            tp,
            fp,
            fn,
        )

        metrics[
            f"{kind}_precision"
        ] = precision

        metrics[
            f"{kind}_recall"
        ] = recall

        metrics[
            f"{kind}_f1"
        ] = f1

        metrics[
            f"{kind}_tp"
        ] = tp

        metrics[
            f"{kind}_fp"
        ] = fp

        metrics[
            f"{kind}_fn"
        ] = fn

    return metrics

def shard_data(
    data,
    rank,
    world_size,
):

    return [
        {
            "index": i,
            "example": example,
        }
        for i, example in enumerate(data)
        if i % world_size == rank
    ]

def run_local_inference(
    model,
    tokenizer,
    device,
    local_items,
):

    local_predictions = []

    total = len(local_items)


    for start in range(
        0,
        total,
        INFERENCE_BATCH_SIZE,
    ):

        batch_items = local_items[
            start:
            start + INFERENCE_BATCH_SIZE
        ]

        batch_examples = [
            item["example"]
            for item in batch_items
        ]

        input_texts = [
            ex["input"]
            for ex in batch_examples
        ]


        raw_outputs = generate_batch(
            model,
            tokenizer,
            device,
            input_texts,
        )


        for item, raw_output in zip(
            batch_items,
            raw_outputs,
        ):

            parsed, valid = (
                parse_prediction(
                    raw_output
                )
            )

            local_predictions.append(
                {
                    "index": item["index"],
                    "input":
                        item["example"]["input"],
                    "gold":
                        item["example"]["output"],
                    "raw_prediction":
                        raw_output,
                    "parsed_prediction":
                        parsed
                        if valid
                        else None,
                    "valid_json":
                        valid,
                }
            )


        print(
            f"[Rank {get_process_rank()}] "
            f"Processed "
            f"{min(start + INFERENCE_BATCH_SIZE, total)}"
            f"/{total}",
            flush=True,
        )


    return local_predictions

def gather_predictions(
    local_predictions,
):

    if not (
        dist.is_available()
        and dist.is_initialized()
    ):

        return local_predictions


    world_size = dist.get_world_size()

    gathered = [
        None
        for _ in range(world_size)
    ]


    dist.all_gather_object(
        gathered,
        local_predictions,
    )


    all_predictions = []

    for part in gathered:

        if part is not None:
            all_predictions.extend(
                part
            )


    all_predictions.sort(
        key=lambda x: x["index"]
    )

    return all_predictions

def print_metrics(
    metrics,
):

    print()
    print("=" * 86)
    print("3-field strict extraction evaluation")
    print("=" * 86)

    print(
        f"{'Level':<16}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'TP':>10}"
        f"{'FP':>10}"
        f"{'FN':>10}"
    )

    print("-" * 86)


    display_names = [
        ("source", "Source"),
        ("prototype", "Prototype"),
        ("application", "Application"),
        ("triplet", "Triplet"),
    ]


    for key, label in display_names:

        print(
            f"{label:<16}"
            f"{metrics[f'{key}_precision']:>12.4f}"
            f"{metrics[f'{key}_recall']:>12.4f}"
            f"{metrics[f'{key}_f1']:>12.4f}"
            f"{metrics[f'{key}_tp']:>10}"
            f"{metrics[f'{key}_fp']:>10}"
            f"{metrics[f'{key}_fn']:>10}"
        )


    print("-" * 86)

    print(
        f"Valid JSON: "
        f"{metrics['valid_json']:.4f}"
    )

    print("=" * 86)

def main():

    try:

        set_seed(SEED)

        distributed = (
            setup_distributed()
        )

        rank = get_process_rank()
        world_size = get_world_size()
        local_rank = get_local_rank()

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is required."
            )

        torch.cuda.set_device(
            local_rank
        )

        device = torch.device(
            f"cuda:{local_rank}"
        )

        if rank == 0:

            print(
                f"Process rank   : {rank}"
            )

            print(
                f"World size    : {world_size}"
            )

            print(
                f"Local rank    : {local_rank}"
            )

            print(
                f"Model path    : {MODEL_PATH}"
            )

            print(
                f"Adapter path  : {ADAPTER_PATH}"
            )

            print(
                f"Test file     : {TEST_FILE}"
            )

            print(
                f"Output dir    : {OUTPUT_DIR}"
            )

            print(
                f"Max length    : {MAX_LENGTH}"
            )

            print(
                f"Max new tok   : {MAX_NEW_TOKENS}"
            )

            print(
                f"Batch/GPU     : {INFERENCE_BATCH_SIZE}"
            )

        test_data = load_json_file(
            TEST_FILE
        )

        validate_dataset(
            test_data,
            TEST_FILE,
        )


        if rank == 0:

            print(
                f"Test samples  : "
                f"{len(test_data)}"
            )

        model, tokenizer = load_model(
            MODEL_PATH,
            ADAPTER_PATH,
            device,
        )

        local_items = shard_data(
            test_data,
            rank,
            world_size,
        )


        print(
            f"[Rank {rank}] "
            f"Local samples: "
            f"{len(local_items)}",
            flush=True,
        )

        local_predictions = (
            run_local_inference(
                model,
                tokenizer,
                device,
                local_items,
            )
        )


        if distributed:

            dist.barrier()

        all_predictions = (
            gather_predictions(
                local_predictions
            )
        )

        if rank == 0:

            prediction_texts = [
                item["raw_prediction"]
                for item in all_predictions
            ]


            metrics = (
                compute_extraction_metrics(
                    test_data,
                    prediction_texts,
                )
            )


            print_metrics(
                metrics
            )

            os.makedirs(
                OUTPUT_DIR,
                exist_ok=True,
            )


            with open(
                PREDICTIONS_FILE,
                "w",
                encoding="utf-8",
            ) as writer:

                for item in all_predictions:

                    writer.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                    )

                    writer.write("\n")


            print()
            print(
                f"Predictions saved to: "
                f"{PREDICTIONS_FILE}"
            )


        if distributed:

            dist.barrier()


    except Exception:

        print()
        print("=" * 72)
        print(
            f"FATAL ERROR ON RANK "
            f"{get_process_rank()}"
        )
        print("=" * 72)

        traceback.print_exc()

        raise


    finally:

        cleanup_distributed()


if __name__ == "__main__":
    main()