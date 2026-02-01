#!/usr/bin/env python3
"""
SLM Function-Calling Benchmark Program (single-file)

Implements a fair 3-way comparison for SLMs:
  1) Zero-shot
  2) Few-shot in-context (k=5 default)
  3) Fine-tuned (LoRA/QLoRA via PEFT)

Primary dataset:
  - NousResearch/hermes-function-calling-v1 (train + eval)

Additional eval datasets:
  - Provide as JSONL via --extra-eval-jsonl (since sources like Berkeley leaderboard + arXiv testbeds
    are not guaranteed to be available as HF datasets in a stable schema).

Training-only dataset for fine-tuning:
  - Provide as HF dataset name or JSONL via --train-dataset / --train-jsonl
    (intended for arXiv:2406.18518 data per your plan).

Metrics (core tables):
  - Schema validity
  - Exact match
  - Function name accuracy
  - Argument-level F1
  - Critical-argument accuracy (defaults to required args; configurable)

Analysis outputs (figures + CSVs):
  - Error taxonomy breakdown
  - Performance vs data size (optional sweep)
  - Cost per correct output (estimated from runtime + GPU hourly cost)

Prompting protocol is enforced:
  - Same system + task prompt for all models
  - Few-shot examples inserted only in few-shot regime
  - Fine-tuned eval uses zero-shot prompt format (no examples)

Decoding parameters (enforced):
  - temperature=0.0
  - top_p=1.0

Notes:
  - This file avoids model-specific prompt tuning.
  - JSON/schema enforcement is done via strict parsing + jsonschema validation. If you want hard
    constrained decoding, integrate a constrained decoding lib; this program keeps fairness and
    simplicity.

Dependencies (typical):
  pip install transformers datasets accelerate peft bitsandbytes jsonschema numpy pandas matplotlib

Example usage:
  # Zero-shot eval on hermes eval split
  python slm_fc_benchmark.py eval \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --primary-dataset NousResearch/hermes-function-calling-v1 \
    --primary-eval-split test \
    --regime zero

  # Few-shot eval (k=5 from primary train split)
  python slm_fc_benchmark.py eval \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --primary-dataset NousResearch/hermes-function-calling-v1 \
    --primary-eval-split test \
    --regime few --k 5

  # Fine-tune (LoRA) then eval
  python slm_fc_benchmark.py finetune \
    --base-model google/gemma-2-9b-it \
    --train-jsonl /path/train_only_fc.jsonl \
    --output-dir ./ft_gemma2_9b_fc_lora \
    --lora \
    --max-train-samples 50000

  python slm_fc_benchmark.py eval \
    --model ./ft_gemma2_9b_fc_lora \
    --primary-dataset NousResearch/hermes-function-calling-v1 \
    --primary-eval-split test \
    --regime zero

"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from datasets import load_dataset, Dataset, DatasetDict
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)

# PEFT is optional until finetune is used
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

# matplotlib only for figures
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ----------------------------
# Prompt protocol (fixed)
# ----------------------------

SYSTEM_PROMPT = (
    "You are an AI assistant that generates structured function calls.\n"
    "You must follow all instructions exactly and output only valid JSON that conforms to the specified function schema.\n"
    "Do not include explanations, comments, or any text outside the JSON output."
)

TASK_PROMPT_TEMPLATE = (
    "You are given a user request and a set of available functions.\n"
    "Your task is to select the single most appropriate function and produce a JSON object that represents a valid call to that function.\n"
    "Follow these rules exactly:\n"
    "- Output only a single JSON object and nothing else.\n"
    "- Do not include explanations, comments, or natural language.\n"
    "- The JSON must strictly conform to the provided function schema.\n"
    "- All required arguments must be included.\n"
    "- Argument names and types must exactly match the schema.\n"
    "- Argument values must be inferred only from the user request.\n"
    "- Do not hallucinate arguments that are not supported by the request.\n"
    "- If multiple functions appear relevant, select the best matching one.\n"
    "Available Functions:\n"
    "{FUNCTION_SCHEMA}\n"
    "User Request:\n"
    "{USER_QUERY}\n"
)

FEWSHOT_EXAMPLE_TEMPLATE = (
    "Example User Request:\n"
    " {EXAMPLE_QUERY}\n"
    "Correct Function Call:\n"
    "{EXAMPLE_JSON_OUTPUT}\n"
)


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first top-level JSON object from arbitrary text.
    This is deliberately strict: if model outputs anything besides JSON, we still attempt to recover
    the first valid object; failures count as invalid_json.
    """
    text = text.strip()
    if not text:
        return None

    # Fast path
    if text.startswith("{") and text.endswith("}"):
        if safe_json_loads(text) is not None:
            return text

    # Bracket matching scan
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = text[start : i + 1].strip()
                if safe_json_loads(cand) is not None:
                    return cand
                # continue searching for later object
                start2 = text.find("{", i + 1)
                if start2 == -1:
                    return None
                start = start2
                depth = 0
    return None


def normalize_call_obj(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize various common function-call JSON shapes into:
      {"name": <str>, "arguments": <dict>}
    Returns None if shape is unrecognized.
    """
    if not isinstance(obj, dict):
        return None

    # Common shapes:
    # 1) {"name": "...", "arguments": {...}}
    if "name" in obj and "arguments" in obj and isinstance(obj["arguments"], dict):
        return {"name": obj["name"], "arguments": obj["arguments"]}

    # 2) {"function": "...", "arguments": {...}} or {"function": {"name":..., "arguments":...}}
    if "function" in obj:
        fn = obj["function"]
        if isinstance(fn, str) and "arguments" in obj and isinstance(obj["arguments"], dict):
            return {"name": fn, "arguments": obj["arguments"]}
        if isinstance(fn, dict):
            name = fn.get("name") or fn.get("function") or fn.get("tool_name")
            args = fn.get("arguments") or fn.get("args") or fn.get("parameters")
            if isinstance(name, str) and isinstance(args, dict):
                return {"name": name, "arguments": args}

    # 3) OpenAI-ish: {"tool_name": "...", "tool_arguments": {...}}
    if "tool_name" in obj and ("tool_arguments" in obj or "arguments" in obj):
        args = obj.get("tool_arguments", obj.get("arguments"))
        if isinstance(args, dict):
            return {"name": obj["tool_name"], "arguments": args}

    # 4) {"name": "...", "parameters": {...}}
    if "name" in obj and "parameters" in obj and isinstance(obj["parameters"], dict):
        return {"name": obj["name"], "arguments": obj["parameters"]}

    return None


def flatten_args(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested dict arguments into dot-path keys for F1 comparisons.
    Lists are kept as-is (treated as atomic).
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_args(v, key))
        else:
            out[key] = v
    return out


def arg_f1(pred_args: Dict[str, Any], gold_args: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Argument-level F1 based on exact key-value matches on flattened args.
    """
    p = flatten_args(pred_args)
    g = flatten_args(gold_args)

    pred_items = set((k, json.dumps(v, sort_keys=True, ensure_ascii=False)) for k, v in p.items())
    gold_items = set((k, json.dumps(v, sort_keys=True, ensure_ascii=False)) for k, v in g.items())

    tp = len(pred_items & gold_items)
    fp = len(pred_items - gold_items)
    fn = len(gold_items - pred_items)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# ----------------------------
# Dataset ingestion (robust)
# ----------------------------

@dataclass
class FunctionCallExample:
    uid: str
    user_query: str
    function_schema_json: Dict[str, Any]  # JSON schema for the expected call (single-call schema)
    gold_call_json: Dict[str, Any]        # gold function call object (as emitted)


def _guess_user_query(sample: Dict[str, Any]) -> Optional[str]:
    # Common keys
    for k in ["user", "query", "instruction", "prompt", "input", "user_query", "question"]:
        if k in sample and isinstance(sample[k], str) and sample[k].strip():
            return sample[k].strip()

    # Conversations
    conv = sample.get("conversation") or sample.get("messages") or sample.get("chat")
    if isinstance(conv, list):
        # Try last user message
        for msg in reversed(conv):
            if isinstance(msg, dict):
                role = (msg.get("role") or msg.get("from") or "").lower()
                content = msg.get("content") or msg.get("value") or msg.get("text")
                if role in ("user", "human") and isinstance(content, str) and content.strip():
                    return content.strip()
    return None


def _guess_gold_call(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Common keys
    for k in ["output", "answer", "expected", "gold", "target", "function_call", "tool_call", "json"]:
        if k in sample:
            v = sample[k]
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                s = extract_first_json_object(v) or v.strip()
                obj = safe_json_loads(s)
                if isinstance(obj, dict):
                    return obj

    # Some datasets store tool calls in messages
    conv = sample.get("conversation") or sample.get("messages") or sample.get("chat")
    if isinstance(conv, list):
        for msg in reversed(conv):
            if isinstance(msg, dict):
                role = (msg.get("role") or msg.get("from") or "").lower()
                # assistant tool call might be in content or function_call
                for kk in ["function_call", "tool_call", "content", "value", "text"]:
                    if kk in msg:
                        v = msg[kk]
                        if isinstance(v, dict):
                            return v
                        if isinstance(v, str):
                            s = extract_first_json_object(v)
                            if s:
                                obj = safe_json_loads(s)
                                if isinstance(obj, dict):
                                    return obj
                if role in ("assistant", "model"):
                    # continue scanning
                    continue
    return None


def _guess_function_schema(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Often tools/functions are provided as list and schema as JSON schema for a call
    # Here we expect a *call schema* (what output JSON must conform to).
    for k in ["function_schema", "schema", "output_schema", "json_schema", "call_schema"]:
        if k in sample:
            v = sample[k]
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                obj = safe_json_loads(v)
                if isinstance(obj, dict):
                    return obj

    # Some datasets store "tools" list of function definitions; we can build a minimal call schema:
    # {
    #   "type": "object",
    #   "properties": {"name": {"enum":[...]},"arguments":{"type":"object"}},
    #   "required":["name","arguments"]
    # }
    tools = sample.get("tools") or sample.get("functions") or sample.get("available_functions")
    if isinstance(tools, list) and tools:
        names = []
        for t in tools:
            if isinstance(t, dict):
                nm = t.get("name") or t.get("function", {}).get("name")
                if isinstance(nm, str):
                    names.append(nm)
        if names:
            return {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": sorted(set(names))},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": True,
            }

    return None


def load_examples_from_hf(
    dataset_name: str,
    split: str,
    limit: Optional[int] = None,
) -> List[FunctionCallExample]:
    ds_all = load_dataset(dataset_name, split="train")

    # Support pseudo-splits when dataset only has train:
    # split can be: "train", "eval", or "train[:80%]" style is handled by HF if available.
    if split == "train":
        ds = ds_all
    elif split in ("eval", "validation", "val"):
        ds = ds_all.train_test_split(test_size=0.2, seed=42)["test"]
    else:
        # fall back to HF split syntax if provided
        ds = load_dataset(dataset_name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    examples: List[FunctionCallExample] = []
    for i, sample in enumerate(ds):
        uid = str(sample.get("id") or sample.get("uid") or f"{dataset_name}:{split}:{i}")

        user_query = _guess_user_query(sample)
        gold_call = _guess_gold_call(sample)
        schema = _guess_function_schema(sample)

        if not user_query or not gold_call or not schema:
            # Skip unparseable samples (keeps evaluation honest: we do not “invent” ground truth)
            continue

        examples.append(
            FunctionCallExample(
                uid=uid,
                user_query=user_query,
                function_schema_json=schema,
                gold_call_json=gold_call,
            )
        )
    return examples


def load_examples_from_jsonl(path: Union[str, Path], limit: Optional[int] = None) -> List[FunctionCallExample]:
    """
    JSONL format expected per line:
      {
        "id": "...",
        "user_query": "...",
        "function_schema": {...},
        "gold_call": {...}
      }
    Keys are flexible; we reuse the same guessers as HF.
    """
    path = Path(path)
    examples: List[FunctionCallExample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and len(examples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            uid = str(sample.get("id") or sample.get("uid") or f"{path.name}:{i}")
            user_query = sample.get("user_query") or _guess_user_query(sample)
            gold_call = sample.get("gold_call") or _guess_gold_call(sample)
            schema = sample.get("function_schema") or _guess_function_schema(sample)
            if not user_query or not gold_call or not schema:
                continue
            examples.append(
                FunctionCallExample(
                    uid=uid,
                    user_query=str(user_query),
                    function_schema_json=schema,
                    gold_call_json=gold_call,
                )
            )
    return examples


# ----------------------------
# Few-shot selection (fixed across models)
# ----------------------------

def select_fewshot_examples(
    train_examples: List[FunctionCallExample],
    k: int,
    seed: int,
) -> List[FunctionCallExample]:
    rng = random.Random(seed)
    if len(train_examples) <= k:
        return train_examples[:]  # deterministic
    idx = list(range(len(train_examples)))
    rng.shuffle(idx)
    return [train_examples[i] for i in idx[:k]]


def render_fewshot_block(examples: List[FunctionCallExample]) -> str:
    blocks = []
    for ex in examples:
        blocks.append(
            FEWSHOT_EXAMPLE_TEMPLATE.format(
                EXAMPLE_QUERY=ex.user_query,
                EXAMPLE_JSON_OUTPUT=json.dumps(ex.gold_call_json, ensure_ascii=False, sort_keys=True),
            )
        )
    return "\n".join(blocks).strip() + "\n"


def render_prompt(
    function_schema_json: Dict[str, Any],
    user_query: str,
    regime: str,
    fewshot_examples: Optional[List[FunctionCallExample]] = None,
) -> str:
    schema_str = json.dumps(function_schema_json, ensure_ascii=False, sort_keys=True)
    base = TASK_PROMPT_TEMPLATE.format(FUNCTION_SCHEMA=schema_str, USER_QUERY=user_query).strip()

    if regime == "zero":
        return f"{SYSTEM_PROMPT}\n\n{base}\n"
    if regime == "few":
        if not fewshot_examples:
            raise ValueError("few-shot regime requires fewshot_examples")
        few = render_fewshot_block(fewshot_examples)
        return f"{SYSTEM_PROMPT}\n\n{base}\n\n{few}"
    raise ValueError(f"Unknown regime: {regime}")


# ----------------------------
# Model runner (HF local)
# ----------------------------

@dataclass
class GenerationStats:
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float


class HFLocalGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_new_tokens: int = 256,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens

        kwargs = {}
        if torch_dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        if load_in_8bit:
            kwargs["load_in_8bit"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            **kwargs,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str) -> Tuple[str, GenerationStats]:
        t0 = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attn = inputs["attention_mask"].to(self.model.device)

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=False,             # temperature=0.0 equivalent
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        t1 = time.time()

        gen_ids = out[0]
        prompt_len = input_ids.shape[-1]
        completion_ids = gen_ids[prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        stats = GenerationStats(
            prompt_tokens=int(prompt_len),
            completion_tokens=int(completion_ids.shape[-1]),
            wall_seconds=float(t1 - t0),
        )
        return text.strip(), stats


# ----------------------------
# Evaluation
# ----------------------------

@dataclass
class EvalResultRow:
    uid: str
    schema_valid: int
    exact_match: int
    fn_name_correct: int
    arg_precision: float
    arg_recall: float
    arg_f1: float
    critical_arg_acc: float
    error_type: str
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float


def build_validator(schema: Dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema)


def determine_critical_args(schema: Dict[str, Any], override: Optional[List[str]] = None) -> List[str]:
    """
    Default definition: required argument names in the call JSON schema, if present,
    otherwise empty.

    If you provide --critical-args-json, you can override per run.
    """
    if override:
        return list(override)

    # Most call schemas won't specify arguments; but if they do:
    # schema["properties"]["arguments"]["required"] might exist.
    try:
        args_schema = schema.get("properties", {}).get("arguments", {})
        req = args_schema.get("required", [])
        if isinstance(req, list):
            return [x for x in req if isinstance(x, str)]
    except Exception:
        pass
    return []


def critical_arg_accuracy(
    pred_args: Dict[str, Any],
    gold_args: Dict[str, Any],
    critical: List[str],
) -> float:
    if not critical:
        return 1.0  # neutral when undefined
    correct = 0
    total = 0
    for k in critical:
        total += 1
        if k in pred_args and k in gold_args and pred_args[k] == gold_args[k]:
            correct += 1
    return correct / total if total else 1.0


def classify_error(
    extracted_json: Optional[str],
    parsed_obj: Optional[Any],
    schema_ok: bool,
    pred_norm: Optional[Dict[str, Any]],
    gold_norm: Optional[Dict[str, Any]],
    validator_error: Optional[str],
) -> str:
    if extracted_json is None or parsed_obj is None:
        return "invalid_json"
    if not schema_ok:
        # Try to make subcategories
        if validator_error and "required" in validator_error:
            return "missing_required_field"
        if validator_error and "additional properties" in validator_error.lower():
            return "extra_field"
        return "schema_invalid"
    if pred_norm is None or gold_norm is None:
        return "unrecognized_call_shape"
    if pred_norm.get("name") != gold_norm.get("name"):
        return "wrong_function"
    # same function, schema ok, but wrong args/values
    return "argument_mismatch"


def eval_examples(
    generator: HFLocalGenerator,
    examples: List[FunctionCallExample],
    regime: str,
    fewshot_examples: Optional[List[FunctionCallExample]],
    critical_args_override: Optional[List[str]],
    gpu_hourly_cost: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[EvalResultRow] = []
    totals = Counter()
    token_totals = Counter()
    time_totals = 0.0

    for ex in examples:
        prompt = render_prompt(
            function_schema_json=ex.function_schema_json,
            user_query=ex.user_query,
            regime=regime,
            fewshot_examples=fewshot_examples,
        )

        raw_out, stats = generator.generate(prompt)

        extracted = extract_first_json_object(raw_out)
        parsed = safe_json_loads(extracted) if extracted else None

        validator = build_validator(ex.function_schema_json)
        schema_ok = False
        validator_error = None
        if parsed is not None:
            try:
                validator.validate(parsed)
                schema_ok = True
            except ValidationError as ve:
                schema_ok = False
                validator_error = str(ve)

        pred_norm = normalize_call_obj(parsed) if parsed is not None else None
        gold_norm = normalize_call_obj(ex.gold_call_json)

        # Metrics
        schema_valid = int(schema_ok)
        exact_match = 0
        fn_name_correct = 0
        ap = ar = af1 = 0.0
        crit_acc = 0.0

        if pred_norm is not None and gold_norm is not None:
            fn_name_correct = int(pred_norm["name"] == gold_norm["name"])
            exact_match = int(pred_norm == gold_norm)
            ap, ar, af1 = arg_f1(pred_norm["arguments"], gold_norm["arguments"])
            critical = determine_critical_args(ex.function_schema_json, critical_args_override)
            crit_acc = critical_arg_accuracy(pred_norm["arguments"], gold_norm["arguments"], critical)

        err_type = classify_error(extracted, parsed, schema_ok, pred_norm, gold_norm, validator_error)

        rows.append(
            EvalResultRow(
                uid=ex.uid,
                schema_valid=schema_valid,
                exact_match=exact_match,
                fn_name_correct=fn_name_correct,
                arg_precision=float(ap),
                arg_recall=float(ar),
                arg_f1=float(af1),
                critical_arg_acc=float(crit_acc),
                error_type=err_type,
                prompt_tokens=stats.prompt_tokens,
                completion_tokens=stats.completion_tokens,
                wall_seconds=stats.wall_seconds,
            )
        )

        totals["n"] += 1
        totals["schema_valid"] += schema_valid
        totals["exact_match"] += exact_match
        totals["fn_name_correct"] += fn_name_correct
        time_totals += stats.wall_seconds
        token_totals["prompt_tokens"] += stats.prompt_tokens
        token_totals["completion_tokens"] += stats.completion_tokens

    df = pd.DataFrame([dataclasses.asdict(r) for r in rows])

    # Aggregate
    n = max(int(totals["n"]), 1)
    summary = {
        "n": n,
        "schema_valid_rate": totals["schema_valid"] / n,
        "exact_match_rate": totals["exact_match"] / n,
        "function_name_acc": totals["fn_name_correct"] / n,
        "arg_f1_mean": float(df["arg_f1"].mean()) if len(df) else 0.0,
        "critical_arg_acc_mean": float(df["critical_arg_acc"].mean()) if len(df) else 0.0,
        "avg_prompt_tokens": token_totals["prompt_tokens"] / n,
        "avg_completion_tokens": token_totals["completion_tokens"] / n,
        "avg_wall_seconds": time_totals / n,
        "error_taxonomy": dict(df["error_type"].value_counts().to_dict()) if len(df) else {},
    }

    # Cost per correct (estimated): runtime hours * GPU hourly cost / num_correct
    # Default "correct" here uses exact match (strictest). You can change downstream.
    correct = int(totals["exact_match"])
    runtime_hours = time_totals / 3600.0
    total_cost = runtime_hours * gpu_hourly_cost
    summary["estimated_total_cost_usd"] = float(total_cost)
    summary["estimated_cost_per_exact_match_usd"] = float(total_cost / correct) if correct > 0 else float("inf")

    return df, summary


def save_eval_artifacts(
    out_dir: Path,
    df: pd.DataFrame,
    summary: Dict[str, Any],
    prefix: str,
) -> None:
    ensure_dir(out_dir)
    df_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.json"
    taxonomy_path = out_dir / f"{prefix}_taxonomy.csv"

    df.to_csv(df_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)

    tax = pd.DataFrame(
        [{"error_type": k, "count": v} for k, v in sorted(summary.get("error_taxonomy", {}).items())]
    )
    tax.to_csv(taxonomy_path, index=False)

    # Simple figure: error taxonomy bar chart
    if plt is not None and len(tax):
        fig_path = out_dir / f"{prefix}_taxonomy.png"
        plt.figure()
        plt.bar(tax["error_type"], tax["count"])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()


# ----------------------------
# Fine-tuning (LoRA SFT)
# ----------------------------

@dataclass
class SFTExample:
    prompt: str
    target_json: str


def build_sft_examples(
    train_examples: List[FunctionCallExample],
    max_samples: Optional[int],
) -> List[SFTExample]:
    if max_samples is not None:
        train_examples = train_examples[: min(max_samples, len(train_examples))]
    sfts: List[SFTExample] = []
    for ex in train_examples:
        prompt = render_prompt(ex.function_schema_json, ex.user_query, regime="zero", fewshot_examples=None)
        target = json.dumps(ex.gold_call_json, ensure_ascii=False, sort_keys=True)
        sfts.append(SFTExample(prompt=prompt, target_json=target))
    return sfts


class PromptTargetDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[SFTExample], tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.items = items
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        # Concatenate prompt + target; mask prompt tokens in labels
        full = it.prompt + it.target_json
        enc = self.tok(
            full,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]

        # Determine prompt length (tokenize prompt separately with same truncation rule)
        enc_p = self.tok(
            it.prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        p_len = int(enc_p["input_ids"].shape[-1])

        labels = input_ids.clone()
        labels[:p_len] = -100  # mask prompt
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def finetune_lora(
    base_model: str,
    train_sft: List[SFTExample],
    output_dir: Union[str, Path],
    seed: int,
    max_length: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    num_epochs: float,
    fp16: bool,
    bf16: bool,
    load_in_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is required for LoRA finetuning. Install: pip install peft")

    set_seed(seed)
    out = ensure_dir(output_dir)

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {}
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", **model_kwargs)

    if load_in_4bit:
        if prepare_model_for_kbit_training is None:
            raise RuntimeError("peft.prepare_model_for_kbit_training unavailable; update peft")
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    ds = PromptTargetDataset(train_sft, tok, max_length=max_length)

    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        report_to=[],
        fp16=fp16,
        bf16=bf16,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        seed=seed,
        remove_unused_columns=False,
    )

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pad manually
        input_ids = [b["input_ids"] for b in batch]
        attn = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tok.pad_token_id)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()

    # Save merged adapter + tokenizer
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))


# ----------------------------
# Sweep: performance vs data size
# ----------------------------

def run_data_size_sweep(
    base_model: str,
    train_examples: List[FunctionCallExample],
    eval_examples_list: List[FunctionCallExample],
    output_dir: Path,
    sizes: List[int],
    common_eval_kwargs: Dict[str, Any],
    finetune_kwargs: Dict[str, Any],
) -> None:
    """
    Trains separate LoRA adapters for each size and evaluates with zero-shot prompt format.
    """
    ensure_dir(output_dir)
    records = []

    for n in sizes:
        run_dir = output_dir / f"sweep_n{n}"
        ensure_dir(run_dir)

        sft = build_sft_examples(train_examples, max_samples=n)
        ft_dir = run_dir / "finetuned"
        finetune_lora(base_model=base_model, train_sft=sft, output_dir=ft_dir, **finetune_kwargs)

        gen = HFLocalGenerator(str(ft_dir), **common_eval_kwargs)
        df, summ = eval_examples(
            generator=gen,
            examples=eval_examples_list,
            regime="zero",
            fewshot_examples=None,
            critical_args_override=None,
            gpu_hourly_cost=common_eval_kwargs.get("gpu_hourly_cost", 0.0),
        )
        save_eval_artifacts(run_dir, df, summ, prefix=f"eval_n{n}")
        records.append(
            {
                "train_size": n,
                "schema_valid_rate": summ["schema_valid_rate"],
                "exact_match_rate": summ["exact_match_rate"],
                "function_name_acc": summ["function_name_acc"],
                "arg_f1_mean": summ["arg_f1_mean"],
                "critical_arg_acc_mean": summ["critical_arg_acc_mean"],
                "estimated_cost_per_exact_match_usd": summ["estimated_cost_per_exact_match_usd"],
            }
        )

    curve = pd.DataFrame(records).sort_values("train_size")
    curve_path = output_dir / "sweep_curve.csv"
    curve.to_csv(curve_path, index=False)

    if plt is not None and len(curve):
        # Plot exact match vs data size
        fig_path = output_dir / "sweep_exact_match.png"
        plt.figure()
        plt.plot(curve["train_size"], curve["exact_match_rate"], marker="o")
        plt.xlabel("Train samples")
        plt.ylabel("Exact match rate")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SLM function-calling benchmark (single-file).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common dataset args (eval)
    def add_eval_dataset_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--primary-dataset", type=str, default="NousResearch/hermes-function-calling-v1")
        pp.add_argument("--primary-train-split", type=str, default="train")
        pp.add_argument("--primary-eval-split", type=str, default="test")
        pp.add_argument("--primary-limit", type=int, default=None)

        pp.add_argument("--extra-eval-jsonl", type=str, nargs="*", default=[],
                        help="Optional additional eval datasets as JSONL files.")
        pp.add_argument("--extra-eval-limit", type=int, default=None)

        pp.add_argument("--seed", type=int, default=42)
        pp.add_argument("--out-dir", type=str, default=f"./runs_{now_stamp()}")

        pp.add_argument("--critical-args-json", type=str, default=None,
                        help="JSON list of critical argument names. If omitted, required args in schema are used.")

        # Cost estimation knobs
        pp.add_argument("--gpu-hourly-cost", type=float, default=0.0,
                        help="USD/hour used for 'cost per correct' estimates (0 disables cost meaningfully).")

    # Eval
    pe = sub.add_parser("eval", help="Run zero-shot or few-shot evaluation on SLM.")
    pe.add_argument("--model", type=str, required=True, help="HF model name or local path.")
    pe.add_argument("--regime", type=str, choices=["zero", "few"], required=True)
    pe.add_argument("--k", type=int, default=5, help="Few-shot example count (few-shot only).")

    pe.add_argument("--max-new-tokens", type=int, default=256)
    pe.add_argument("--device-map", type=str, default="auto")
    pe.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    pe.add_argument("--load-in-4bit", action="store_true")
    pe.add_argument("--load-in-8bit", action="store_true")

    add_eval_dataset_args(pe)

    # Finetune
    pf = sub.add_parser("finetune", help="Fine-tune SLM with LoRA/QLoRA on training-only dataset.")
    pf.add_argument("--base-model", type=str, required=True)
    pf.add_argument("--output-dir", type=str, required=True)

    # Training data sources
    pf.add_argument("--train-dataset", type=str, default=None, help="HF dataset name (optional).")
    pf.add_argument("--train-split", type=str, default="train")
    pf.add_argument("--train-jsonl", type=str, default=None, help="Training-only JSONL path (optional).")
    pf.add_argument("--max-train-samples", type=int, default=None)

    # LoRA control
    pf.add_argument("--lora", action="store_true", default=True, help="Use LoRA (default on).")
    pf.add_argument("--load-in-4bit", action="store_true", help="QLoRA: load base in 4bit for training.")
    pf.add_argument("--lora-r", type=int, default=16)
    pf.add_argument("--lora-alpha", type=int, default=32)
    pf.add_argument("--lora-dropout", type=float, default=0.05)

    # SFT hyperparams
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--max-length", type=int, default=2048)
    pf.add_argument("--lr", type=float, default=2e-4)
    pf.add_argument("--batch-size", type=int, default=1)
    pf.add_argument("--grad-accum", type=int, default=16)
    pf.add_argument("--epochs", type=float, default=1.0)
    pf.add_argument("--fp16", action="store_true")
    pf.add_argument("--bf16", action="store_true")

    # Sweep
    ps = sub.add_parser("sweep", help="Data-size sweep: fine-tune LoRA at multiple sizes + eval.")
    ps.add_argument("--base-model", type=str, required=True)
    ps.add_argument("--output-dir", type=str, required=True)

    ps.add_argument("--train-dataset", type=str, default=None)
    ps.add_argument("--train-split", type=str, default="train")
    ps.add_argument("--train-jsonl", type=str, default=None)
    ps.add_argument("--sizes", type=str, required=True, help="Comma-separated list, e.g. 1000,5000,20000")

    ps.add_argument("--primary-dataset", type=str, default="NousResearch/hermes-function-calling-v1")
    ps.add_argument("--primary-eval-split", type=str, default="test")
    ps.add_argument("--primary-limit", type=int, default=None)

    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--max-new-tokens", type=int, default=256)
    ps.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ps.add_argument("--load-in-4bit", action="store_true")

    ps.add_argument("--max-train-samples", type=int, default=None)
    ps.add_argument("--max-length", type=int, default=2048)
    ps.add_argument("--lr", type=float, default=2e-4)
    ps.add_argument("--batch-size", type=int, default=1)
    ps.add_argument("--grad-accum", type=int, default=16)
    ps.add_argument("--epochs", type=float, default=1.0)
    ps.add_argument("--fp16", action="store_true")
    ps.add_argument("--bf16", action="store_true")

    ps.add_argument("--lora-r", type=int, default=16)
    ps.add_argument("--lora-alpha", type=int, default=32)
    ps.add_argument("--lora-dropout", type=float, default=0.05)

    ps.add_argument("--gpu-hourly-cost", type=float, default=0.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(getattr(args, "seed", 42))

    if args.cmd == "eval":
        out_dir = ensure_dir(args.out_dir)

        # Load primary eval set (eval-only for evaluation)
        primary_eval = load_examples_from_hf(args.primary_dataset, args.primary_eval_split, limit=args.primary_limit)

        # Load few-shot candidates from primary train split (training-only usage for examples)
        primary_train = load_examples_from_hf(args.primary_dataset, args.primary_train_split, limit=None)

        # Additional eval datasets (JSONL)
        extras = []
        for pth in args.extra_eval_jsonl:
            extras.extend(load_examples_from_jsonl(pth, limit=args.extra_eval_limit))

        eval_set = primary_eval + extras

        fewshot = None
        if args.regime == "few":
            fewshot = select_fewshot_examples(primary_train, k=args.k, seed=args.seed)

        critical_override = None
        if args.critical_args_json:
            critical_override = json.loads(Path(args.critical_args_json).read_text(encoding="utf-8"))
            if not isinstance(critical_override, list) or not all(isinstance(x, str) for x in critical_override):
                raise ValueError("--critical-args-json must be a JSON file containing a list of strings")

        gen = HFLocalGenerator(
            model_name_or_path=args.model,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            max_new_tokens=args.max_new_tokens,
        )

        df, summ = eval_examples(
            generator=gen,
            examples=eval_set,
            regime=args.regime,
            fewshot_examples=fewshot,
            critical_args_override=critical_override,
            gpu_hourly_cost=args.gpu_hourly_cost,
        )

        prefix = f"eval_{args.regime}_{Path(args.model).name}"
        save_eval_artifacts(out_dir, df, summ, prefix=prefix)

        # Console summary (no fluff)
        print(json.dumps(summ, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"\nSaved: {out_dir}")

        return

    if args.cmd == "finetune":
        if not (args.train_dataset or args.train_jsonl):
            raise ValueError("Provide training-only data via --train-dataset or --train-jsonl")

        # Load training-only dataset
        train_examples: List[FunctionCallExample] = []
        if args.train_dataset:
            train_examples = load_examples_from_hf(args.train_dataset, args.train_split, limit=args.max_train_samples)
        else:
            train_examples = load_examples_from_jsonl(args.train_jsonl, limit=args.max_train_samples)

        sft = build_sft_examples(train_examples, max_samples=args.max_train_samples)

        finetune_lora(
            base_model=args.base_model,
            train_sft=sft,
            output_dir=args.output_dir,
            seed=args.seed,
            max_length=args.max_length,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            num_epochs=args.epochs,
            fp16=args.fp16,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        print(f"Fine-tune complete. Saved model to: {args.output_dir}")
        return

    if args.cmd == "sweep":
        if not (args.train_dataset or args.train_jsonl):
            raise ValueError("Provide training-only data via --train-dataset or --train-jsonl")

        sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
        if not sizes:
            raise ValueError("--sizes must be a comma-separated list of ints")

        # Training pool
        if args.train_dataset:
            train_pool = load_examples_from_hf(args.train_dataset, args.train_split, limit=args.max_train_samples)
        else:
            train_pool = load_examples_from_jsonl(args.train_jsonl, limit=args.max_train_samples)

        # Eval set (primary)
        eval_set = load_examples_from_hf(args.primary_dataset, args.primary_eval_split, limit=args.primary_limit)

        out_dir = ensure_dir(args.output_dir)

        common_eval_kwargs = dict(
            device_map="auto",
            torch_dtype=args.torch_dtype,
            load_in_4bit=False,   # eval typically can be fp16/bf16; keep separate from training QLoRA
            load_in_8bit=False,
            max_new_tokens=args.max_new_tokens,
            gpu_hourly_cost=args.gpu_hourly_cost,
        )

        finetune_kwargs = dict(
            seed=args.seed,
            max_length=args.max_length,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            num_epochs=args.epochs,
            fp16=args.fp16,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        run_data_size_sweep(
            base_model=args.base_model,
            train_examples=train_pool,
            eval_examples_list=eval_set,
            output_dir=out_dir,
            sizes=sizes,
            common_eval_kwargs=common_eval_kwargs,
            finetune_kwargs=finetune_kwargs,
        )

        print(f"Sweep complete. Outputs in: {out_dir}")
        return


if __name__ == "__main__":
    main()
