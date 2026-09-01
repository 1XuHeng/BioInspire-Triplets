import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import sys
import json
import time
import math
import random
import traceback
import inspect
import unicodedata
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from datasets import Dataset
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model



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

MODEL_PATH = "./model/llama"
TRAIN_FILE = "./data/train.json"
VALIDATION_FILE = "./data/validation.json"

OUTPUT_DIR = "./outputs/llama/lora"
LOG_DIR = "./logs"
CURVE_DIR = os.path.join(OUTPUT_DIR, "training_curves")

MAX_LENGTH = 2048
SEED = 42


LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

NUM_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH = 2
PER_DEVICE_EVAL_BATCH = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
DATALOADER_NUM_WORKERS = 4
LR_SCHEDULER_TYPE = "cosine"


GENERATION_BATCH_SIZE = 4
GENERATION_MAX_NEW_TOKENS = 384
BEST_MODEL_METRIC = "triplet_f1"


MULTI_RELATION_REPEAT = 1


class TeeStream:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        try:
            return self.terminal.isatty()
        except Exception:
            return False

    def fileno(self):
        return self.terminal.fileno()


def get_process_rank():
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def setup_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    rank = get_process_rank()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = os.path.join(log_dir, f"train_lora_3field_{timestamp}_rank{rank}.log")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    return log_path, log_file, original_stdout, original_stderr


def close_log(log_file, original_stdout, original_stderr):
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    try:
        log_file.flush()
        log_file.close()
    except Exception:
        pass



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: top level must be a JSON array")
    return data


def validate_dataset(data, path):
    for i, item in enumerate(data, start=1):
        if "input" not in item:
            raise ValueError(f"{path}: item {i} missing input")
        if "output" not in item:
            raise ValueError(f"{path}: item {i} missing output")
        if not isinstance(item["input"], str):
            raise ValueError(f"{path}: item {i} input must be a string")
        if not isinstance(item["output"], list):
            raise ValueError(f"{path}: item {i} output must be a list")

        input_text = item["input"]
        for j, relation in enumerate(item["output"], start=1):
            if not isinstance(relation, dict):
                raise ValueError(f"{path}: item {i}, relation {j} must be an object")

            for field in REQUIRED_OUTPUT_FIELDS:
                if field not in relation:
                    raise ValueError(
                        f"{path}: item {i}, relation {j}, missing {field}"
                    )
                if not isinstance(relation[field], str):
                    raise ValueError(
                        f"{path}: item {i}, relation {j}, {field} must be a string"
                    )

            for field in ("source", "prototype", "application"):
                value = relation[field]
                if value not in input_text:
                    raise ValueError(
                        f"{path}: item {i}, relation {j}, {field} is not a "
                        f"contiguous verbatim span of input: {value!r}"
                    )



def _relation_tuple(relation):
    return tuple(relation[field] for field in REQUIRED_OUTPUT_FIELDS)


def canonicalize_output(input_text, relations):
    unique = []
    seen = set()
    for relation in relations:
        normalized = {field: relation[field] for field in REQUIRED_OUTPUT_FIELDS}
        key = _relation_tuple(normalized)
        if key not in seen:
            seen.add(key)
            unique.append(normalized)

    def pos(span):
        p = input_text.find(span)
        return p if p >= 0 else 10**12

    unique.sort(
        key=lambda r: (
            min(pos(r["source"]), pos(r["prototype"]), pos(r["application"])),
            pos(r["source"]),
            pos(r["prototype"]),
            pos(r["application"]),
            r["source"],
            r["prototype"],
            r["application"],
        )
    )
    return unique


def maybe_oversample_multi_relation(data, repeat):
    if repeat <= 1:
        return data

    expanded = []
    for item in data:
        copies = repeat if len(item["output"]) >= 2 else 1
        for _ in range(copies):
            expanded.append(item)
    return expanded


def print_dataset_diagnostics(data, name):
    relation_counts = Counter(len(x["output"]) for x in data)

    print()
    print("=" * 72)
    print(f"{name} dataset diagnostics")
    print("=" * 72)
    print(f"samples: {len(data)}")
    print(f"relations: {sum(k * v for k, v in relation_counts.items())}")
    print(f"relations/sample: {dict(sorted(relation_counts.items()))}")
    print("task fields: source, prototype, application")
    print("type fields in the source JSON are ignored during SFT")
    print("=" * 72)
    print()



def render_chat_template(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def encode_example(example, tokenizer, max_length):
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "A fast tokenizer is required because completion-only masking uses "
            "offset_mapping."
        )

    user_text = example["input"]
    output = canonicalize_output(user_text, example["output"])
    assistant_text = json.dumps(
        output,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    full_text = render_chat_template(
        tokenizer,
        full_messages,
        add_generation_prompt=False,
    )

    assistant_char_start = full_text.rfind(assistant_text)
    if assistant_char_start < 0:
        raise RuntimeError("Could not locate assistant JSON in rendered chat template.")

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]

    if len(input_ids) > max_length:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "valid_target": False,
            "sequence_length": len(input_ids),
        }

    labels = input_ids.copy()
    first_assistant_token = None

    for idx, (char_start, char_end) in enumerate(offsets):
        if char_end <= assistant_char_start or char_start < assistant_char_start:
            labels[idx] = -100
            continue
        first_assistant_token = idx
        break

    if first_assistant_token is None or not any(x != -100 for x in labels):
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "valid_target": False,
            "sequence_length": len(input_ids),
        }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "valid_target": True,
        "sequence_length": len(input_ids),
    }


def build_dataset(raw_data, tokenizer, max_length, name):
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: encode_example(x, tokenizer, max_length),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {name}",
    )

    before = len(dataset)
    lengths = dataset["sequence_length"]
    if lengths:
        print(
            f"{name} sequence length: min={min(lengths)}, max={max(lengths)}, "
            f"mean={sum(lengths) / len(lengths):.1f}"
        )

    dataset = dataset.filter(
        lambda x: x["valid_target"],
        desc=f"Filtering {name}",
    )
    after = len(dataset)
    dropped = before - after
    print(
        f"{name}: kept {after}/{before}; dropped {dropped} samples because "
        f"complete target did not fit MAX_LENGTH={max_length}."
    )

    if after == 0:
        raise RuntimeError(f"{name}: no valid samples remain after filtering.")

    if before > 0 and dropped / before > 0.01:
        print(
            f"WARNING: {100.0 * dropped / before:.2f}% of {name} was dropped. "
            "Consider increasing MAX_LENGTH if GPU memory allows."
        )

    return dataset.remove_columns(["valid_target", "sequence_length"])


class CausalLMDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []
        labels = []

        for feature in features:
            length = len(feature["input_ids"])
            padding_length = max_len - length
            input_ids.append(feature["input_ids"] + [pad_id] * padding_length)
            attention_masks.append(
                feature["attention_mask"] + [0] * padding_length
            )
            labels.append(feature["labels"] + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def normalize_text(value):
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.strip().split())


def validate_prediction_schema(obj):
    if not isinstance(obj, list):
        return False

    for rel in obj:
        if not isinstance(rel, dict):
            return False
        if set(rel.keys()) != set(REQUIRED_OUTPUT_FIELDS):
            return False
        if not all(isinstance(rel[k], str) for k in REQUIRED_OUTPUT_FIELDS):
            return False
    return True


def parse_prediction(text):
    try:
        obj = json.loads(text.strip())
    except Exception:
        return [], False
    if not validate_prediction_schema(obj):
        return [], False
    return obj, True


def _safe_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _relation_key(rel, kind):
    s = normalize_text(rel["source"])
    p = normalize_text(rel["prototype"])
    a = normalize_text(rel["application"])

    if kind == "source":
        return s
    if kind == "prototype":
        return p
    if kind == "application":
        return a
    if kind == "triplet":
        return (s, p, a)
    raise ValueError(kind)


def compute_extraction_metrics(raw_examples, prediction_texts):
    kinds = [
        "source",
        "prototype",
        "application",
        "triplet",
    ]
    counts = {k: {"TP": 0, "FP": 0, "FN": 0} for k in kinds}
    valid_json_count = 0

    for example, pred_text in zip(raw_examples, prediction_texts):
        gold = canonicalize_output(example["input"], example["output"])
        pred, valid = parse_prediction(pred_text)
        valid_json_count += int(valid)

        for kind in kinds:
            gold_set = {_relation_key(rel, kind) for rel in gold}
            pred_set = {_relation_key(rel, kind) for rel in pred}
            counts[kind]["TP"] += len(gold_set & pred_set)
            counts[kind]["FP"] += len(pred_set - gold_set)
            counts[kind]["FN"] += len(gold_set - pred_set)

    metrics = {
        "valid_json": valid_json_count / max(1, len(raw_examples)),
    }

    for kind in kinds:
        tp = counts[kind]["TP"]
        fp = counts[kind]["FP"]
        fn = counts[kind]["FN"]
        precision, recall, f1 = _safe_f1(tp, fp, fn)
        metrics[f"{kind}_precision"] = precision
        metrics[f"{kind}_recall"] = recall
        metrics[f"{kind}_f1"] = f1

    return metrics


class ExtractionTrainer(Trainer):
    def __init__(
        self,
        *args,
        tokenizer_for_generation=None,
        raw_eval_examples=None,
        generation_batch_size=4,
        generation_max_new_tokens=768,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer_for_generation = tokenizer_for_generation
        self.raw_eval_examples = raw_eval_examples or []
        self.generation_batch_size = generation_batch_size
        self.generation_max_new_tokens = generation_max_new_tokens

    def _generate_validation_predictions(self):
        tokenizer = self.tokenizer_for_generation
        if tokenizer is None:
            raise RuntimeError("tokenizer_for_generation is required")

        model = self.accelerator.unwrap_model(self.model)
        device = next(model.parameters()).device
        was_training = model.training
        old_use_cache = getattr(model.config, "use_cache", None)
        old_padding_side = tokenizer.padding_side

        model.eval()
        model.config.use_cache = True
        tokenizer.padding_side = "left"

        predictions = []
        try:
            for start in range(0, len(self.raw_eval_examples), self.generation_batch_size):
                batch_examples = self.raw_eval_examples[
                    start : start + self.generation_batch_size
                ]
                prompts = []
                for ex in batch_examples:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ex["input"]},
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
                batch = {k: v.to(device) for k, v in batch.items()}
                prompt_width = batch["input_ids"].shape[1]

                with torch.inference_mode():
                    eos_token_id = getattr(model.generation_config, "eos_token_id", None)
                    if eos_token_id is None:
                        eos_token_id = getattr(model.config, "eos_token_id", None)
                    if eos_token_id is None:
                        eos_token_id = tokenizer.eos_token_id

                    generated = model.generate(
                        **batch,
                        max_new_tokens=self.generation_max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_token_id,
                    )

                continuations = generated[:, prompt_width:]
                decoded = tokenizer.batch_decode(
                    continuations,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                predictions.extend(decoded)

        finally:
            tokenizer.padding_side = old_padding_side
            if old_use_cache is not None:
                model.config.use_cache = old_use_cache
            if was_training:
                model.train()

        return predictions

    def _distributed_task_metrics(self):
        metrics = None
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0

        if rank == 0:
            prediction_texts = self._generate_validation_predictions()
            metrics = compute_extraction_metrics(
                self.raw_eval_examples,
                prediction_texts,
            )

        if distributed:
            payload = [metrics]
            dist.broadcast_object_list(payload, src=0)
            metrics = payload[0]

        return metrics

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.raw_eval_examples:
            task_metrics = self._distributed_task_metrics()
            prefixed = {
                f"{metric_key_prefix}_{key}": value
                for key, value in task_metrics.items()
            }
            metrics.update(prefixed)
            self.log(prefixed)

            if self.is_world_process_zero():
                print()
                print("=" * 72)
                print("Generation-based validation")
                print("=" * 72)
                print(f"Valid JSON          : {task_metrics['valid_json']:.6f}")
                print(f"Source F1           : {task_metrics['source_f1']:.6f}")
                print(f"Prototype F1        : {task_metrics['prototype_f1']:.6f}")
                print(f"Application F1      : {task_metrics['application_f1']:.6f}")
                print(f"Triplet F1          : {task_metrics['triplet_f1']:.6f}")
                print("=" * 72)
                print()

        return metrics


def calculate_training_steps(num_train_samples, world_size):
    samples_per_rank = math.ceil(num_train_samples / world_size)
    dataloader_steps_per_epoch = math.ceil(
        samples_per_rank / PER_DEVICE_TRAIN_BATCH
    )
    optimizer_steps_per_epoch = math.ceil(
        dataloader_steps_per_epoch / GRADIENT_ACCUMULATION_STEPS
    )
    total_optimizer_steps = math.ceil(optimizer_steps_per_epoch * NUM_EPOCHS)
    warmup_steps = int(round(total_optimizer_steps * WARMUP_RATIO))
    if WARMUP_RATIO > 0 and total_optimizer_steps > 0:
        warmup_steps = max(1, warmup_steps)
    warmup_steps = min(warmup_steps, total_optimizer_steps)
    return {
        "samples_per_rank": samples_per_rank,
        "dataloader_steps_per_epoch": dataloader_steps_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "warmup_steps": warmup_steps,
    }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def collect_lora_stats(model):
    stats = []
    for name, param in model.named_parameters():
        if "lora_B" in name:
            tensor = param.detach().float()
            stats.append(
                {
                    "name": name,
                    "mean_abs": tensor.abs().mean().item(),
                    "norm": tensor.norm().item(),
                }
            )
    return stats


def print_lora_stats(model, max_rows=20):
    stats = collect_lora_stats(model)
    print()
    print("=" * 72)
    print("LoRA B-matrix diagnostics")
    print("=" * 72)
    if not stats:
        print("No lora_B parameters found.")
        return
    for item in stats[:max_rows]:
        print(
            f"{item['name']} | mean_abs={item['mean_abs']:.8f} | "
            f"norm={item['norm']:.8f}"
        )
    if len(stats) > max_rows:
        print(f"... {len(stats) - max_rows} more lora_B tensors")
    print("=" * 72)
    print()


def print_cuda_memory(prefix="CUDA memory"):
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    gib = 1024**3
    print(
        f"{prefix} | rank={get_process_rank()} | free={free_bytes / gib:.2f} GiB "
        f"| total={total_bytes / gib:.2f} GiB | allocated={allocated / gib:.2f} GiB "
        f"| reserved={reserved / gib:.2f} GiB"
    )


class TrainInfoCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        world_size = getattr(args, "world_size", 1)
        global_batch = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * world_size
        )
        print()
        print("=" * 72)
        print("Llama-3.1-8B-Instruct 3-field LoRA SFT started")
        print("=" * 72)
        print(f"Epochs                 : {args.num_train_epochs}")
        print(f"World size             : {world_size}")
        print(f"Per-device batch       : {args.per_device_train_batch_size}")
        print(f"Gradient accumulation  : {args.gradient_accumulation_steps}")
        print(f"Effective global batch : {global_batch}")
        print(f"Learning rate          : {args.learning_rate}")
        print(f"Warmup steps           : {args.warmup_steps}")
        print(f"Trainer max steps      : {state.max_steps}")
        print(f"Best metric            : {BEST_MODEL_METRIC}")
        print("=" * 72)
        print()


def save_training_curves(log_history, curve_dir):
    os.makedirs(curve_dir, exist_ok=True)
    history_path = os.path.join(curve_dir, "trainer_log_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    def plot_series(x, y, xlabel, ylabel, title, filename):
        if not x:
            return
        plt.figure(figsize=(9, 6))
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        path = os.path.join(curve_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    triplet_steps, triplet_f1 = [], []

    for rec in log_history:
        step = rec.get("step")
        if step is None:
            continue
        if "loss" in rec:
            train_steps.append(step)
            train_losses.append(rec["loss"])
        if "eval_loss" in rec:
            eval_steps.append(step)
            eval_losses.append(rec["eval_loss"])
        if "eval_triplet_f1" in rec:
            triplet_steps.append(step)
            triplet_f1.append(rec["eval_triplet_f1"])

    if train_steps or eval_steps:
        plt.figure(figsize=(9, 6))
        if train_steps:
            plt.plot(train_steps, train_losses, label="Train loss")
        if eval_steps:
            plt.plot(eval_steps, eval_losses, marker="o", label="Validation loss")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        path = os.path.join(curve_dir, "loss_curve.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    plot_series(
        triplet_steps,
        triplet_f1,
        "Training step",
        "Triplet F1",
        "Validation Triplet F1",
        "triplet_f1_curve.png",
    )


@record
def main():
    log_path, log_file, original_stdout, original_stderr = setup_log(LOG_DIR)
    training_start_time = time.time()

    try:
        set_seed(SEED)

        print(f"Process rank   : {get_process_rank()}")
        print(f"World size    : {get_world_size()}")
        print(f"Model path    : {MODEL_PATH}")
        print(f"Train file    : {TRAIN_FILE}")
        print(f"Validation    : {VALIDATION_FILE}")
        print(f"Output dir    : {OUTPUT_DIR}")
        print(f"Log file      : {log_path}")
        print(f"Max length    : {MAX_LENGTH}")
        print("Chat template  : Meta-Llama-3.1-Instruct native template")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        if torch.cuda.is_available():
            use_bf16 = torch.cuda.is_bf16_supported()
            model_dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            use_bf16 = False
            model_dtype = torch.float32
        use_fp16 = torch.cuda.is_available() and not use_bf16

        print(f"Model dtype    : {model_dtype}")
        print(f"bf16           : {use_bf16}")
        print(f"fp16           : {use_fp16}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

        total_params, trainable_params = count_parameters(model)
        print()
        print("=" * 72)
        print("Parameter diagnostics")
        print("=" * 72)
        print(f"Total parameters     : {total_params:,}")
        print(f"Trainable parameters : {trainable_params:,}")
        print(f"Trainable percent    : {100.0 * trainable_params / total_params:.6f}%")
        print("=" * 72)
        model.print_trainable_parameters()
        print_lora_stats(model)
        print_cuda_memory("After LoRA model load")

        train_raw = load_json_file(TRAIN_FILE)
        validation_raw = load_json_file(VALIDATION_FILE)
        validate_dataset(train_raw, TRAIN_FILE)
        validate_dataset(validation_raw, VALIDATION_FILE)

        print_dataset_diagnostics(train_raw, "train")
        print_dataset_diagnostics(validation_raw, "validation")

        train_raw_for_training = maybe_oversample_multi_relation(
            train_raw,
            MULTI_RELATION_REPEAT,
        )
        if len(train_raw_for_training) != len(train_raw):
            print(
                f"Multi-relation oversampling: {len(train_raw)} -> "
                f"{len(train_raw_for_training)} samples"
            )

        train_dataset = build_dataset(
            train_raw_for_training,
            tokenizer,
            MAX_LENGTH,
            name="train",
        )
        eval_dataset = build_dataset(
            validation_raw,
            tokenizer,
            MAX_LENGTH,
            name="validation",
        )
        data_collator = CausalLMDataCollator(tokenizer)

        step_info = calculate_training_steps(
            num_train_samples=len(train_dataset),
            world_size=get_world_size(),
        )

        if get_process_rank() == 0:
            print()
            print("=" * 72)
            print("Training-step calculation")
            print("=" * 72)
            for key, value in step_info.items():
                print(f"{key:28s}: {value}")
            if step_info["total_optimizer_steps"] < 150:
                print(
                    "WARNING: fewer than 150 optimizer updates. If validation "
                    "F1 is still rising at the end, reduce effective batch or "
                    "increase epochs rather than only increasing LoRA rank."
                )
            print("=" * 72)
            print()

        training_arg_kwargs = {
            "output_dir": OUTPUT_DIR,
            "num_train_epochs": NUM_EPOCHS,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH,
            "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": step_info["warmup_steps"],
            "lr_scheduler_type": LR_SCHEDULER_TYPE,
            "optim": "adamw_torch",
            "bf16": use_bf16,
            "fp16": use_fp16,
            "gradient_checkpointing": True,
            "max_grad_norm": MAX_GRAD_NORM,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": LOGGING_STEPS,
            "load_best_model_at_end": True,
            "metric_for_best_model": BEST_MODEL_METRIC,
            "greater_is_better": True,
            "save_total_limit": 3,
            "seed": SEED,
            "data_seed": SEED,
            "dataloader_num_workers": DATALOADER_NUM_WORKERS,
            "remove_unused_columns": False,
            "report_to": "none",
            "ddp_find_unused_parameters": False,
            "group_by_length": True,
        }

        ta_signature = inspect.signature(TrainingArguments.__init__)
        supported_ta_args = set(ta_signature.parameters)

        if (
            "eval_strategy" in training_arg_kwargs
            and "eval_strategy" not in supported_ta_args
            and "evaluation_strategy" in supported_ta_args
        ):
            training_arg_kwargs["evaluation_strategy"] = (
                training_arg_kwargs.pop("eval_strategy")
            )

        unsupported_ta_args = sorted(
            key
            for key in training_arg_kwargs
            if key not in supported_ta_args
        )
        for key in unsupported_ta_args:
            training_arg_kwargs.pop(key)

        if get_process_rank() == 0:
            print()
            print("=" * 72)
            print("TrainingArguments compatibility")
            print("=" * 72)
            if unsupported_ta_args:
                print(
                    "Unsupported optional arguments skipped: "
                    + ", ".join(unsupported_ta_args)
                )
            else:
                print("All requested TrainingArguments are supported.")
            print("=" * 72)
            print()

        training_args = TrainingArguments(**training_arg_kwargs)

        trainer = ExtractionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[TrainInfoCallback()],
            tokenizer_for_generation=tokenizer,
            raw_eval_examples=validation_raw,
            generation_batch_size=GENERATION_BATCH_SIZE,
            generation_max_new_tokens=GENERATION_MAX_NEW_TOKENS,
        )

        train_result = trainer.train()
        results = trainer.evaluate()

        final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
        if trainer.is_world_process_zero():
            os.makedirs(final_dir, exist_ok=True)
            print("LoRA statistics AFTER training:")
            print_lora_stats(trainer.model)
            trainer.model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            save_training_curves(trainer.state.log_history, CURVE_DIR)

            world_size = training_args.world_size
            effective_global_batch = (
                PER_DEVICE_TRAIN_BATCH
                * GRADIENT_ACCUMULATION_STEPS
                * world_size
            )

            config = {
                "base_model": MODEL_PATH,
                "training_method": "LoRA SFT + generation-based model selection",
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "lora_target_modules": LORA_TARGET_MODULES,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "trainable_percent": 100.0 * trainable_params / total_params,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "lr_scheduler_type": LR_SCHEDULER_TYPE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio_requested": WARMUP_RATIO,
                "warmup_steps": step_info["warmup_steps"],
                "estimated_total_optimizer_steps": step_info["total_optimizer_steps"],
                "num_train_epochs": NUM_EPOCHS,
                "max_sequence_length": MAX_LENGTH,
                "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH,
                "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "world_size": world_size,
                "effective_global_batch_size": effective_global_batch,
                "generation_batch_size": GENERATION_BATCH_SIZE,
                "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS,
                "metric_for_best_model": BEST_MODEL_METRIC,
                "multi_relation_repeat": MULTI_RELATION_REPEAT,
                "global_step": trainer.state.global_step,
                "seed": SEED,
                "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
                "loss": "assistant/completion only",
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "train_metrics": train_result.metrics,
                "final_eval_metrics": results,
                "log_file": log_path,
                "curve_dir": CURVE_DIR,
                "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            }

            config_path = os.path.join(final_dir, "training_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            elapsed_seconds = time.time() - training_start_time
            print()
            print("=" * 72)
            print("Llama-3.1-8B-Instruct 3-field LoRA training completed")
            print("=" * 72)
            print(f"Adapter saved to       : {final_dir}")
            print(f"Best checkpoint        : {trainer.state.best_model_checkpoint}")
            print(f"Best task metric       : {trainer.state.best_metric}")
            print(f"Global optimizer steps : {trainer.state.global_step}")
            print(f"Training time          : {elapsed_seconds / 60:.2f} min")
            print(f"Training curves        : {CURVE_DIR}")
            print(f"Log file               : {log_path}")
            print("=" * 72)

    except Exception:
        print()
        print("=" * 72)
        print(f"FATAL ERROR ON RANK {get_process_rank()}")
        print("=" * 72)
        traceback.print_exc()
        print("=" * 72)
        raise

    finally:
        close_log(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()