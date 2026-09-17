#!/usr/bin/env python3
"""
Phase 1 — Golden Record Discovery.

Spec: `Attention Calibration Distillation — Three-Phase Experimental Project
Specification.md`, sections 11-18, 41, 44, 51, 54, 55.

What this file does
-------------------
1. Loads ProofWriter CWA (depth-2) from the HuggingFace Hub.
2. Selects a deterministic pool of 500 target questions, balanced 250 True /
   250 False, keeping only questions with QDep >= 1 (spec section 51).
3. Builds two fixed, verified demonstrations from dataset proofs, taken from
   theories that are excluded from the target pool (spec sections 7, 8).
4. Runs every target under two conditions with the SAME model and the SAME
   chat template:
     - zero-shot Student : target context + target question
     - two-shot Teacher  : 2 demonstrations + target context + target question
   The target's gold answer and gold proof are NEVER placed in either prompt
   (spec section 4, section 44 Rule 2).
5. Parses the final answer, compares against the gold label, classifies, and
   writes every required artifact.

This is the only executable Python file for Phase 1 (spec section 55).

Stages
------
    prepare   -> pool + demonstrations + phase1_config.json
    generate  -> raw zero-shot and few-shot generations (append-only cache)
    classify  -> all_records.jsonl and the four derived category files
    all       -> prepare, generate, classify

Every stage is resumable. Raw generation files are append-only and are never
truncated (spec section 44 Rule 6).

Usage
-----
    # Local: build the pool and demonstrations only (no GPU needed)
    python phase1.py --stage prepare

    # Colab: full run, artifacts persisted to Drive
    export ACD_ROOT=/content/drive/MyDrive/acd02
    python phase1.py --stage all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"

log = logging.getLogger("phase1")


# ---------------------------------------------------------------------------
# Configuration and paths
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        sys.exit(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def artifact_root(cfg: dict) -> Path:
    """Artifact root: $ACD_ROOT when set, else the configured default.

    Colab's local filesystem is temporary (spec section 54), so in Colab
    ACD_ROOT must point at mounted Google Drive.
    """
    env_var = cfg["paths"]["root_env_var"]
    raw = os.environ.get(env_var) or cfg["paths"]["default_root"]
    return Path(raw).expanduser().resolve()


def artifact_paths(cfg: dict) -> dict:
    root = artifact_root(cfg)
    out = {}
    for key, rel in cfg["paths"].items():
        if key in ("root_env_var", "default_root"):
            continue
        out[key] = (root / rel).resolve()
    out["_root"] = root
    return out


def ensure_dirs(paths: dict) -> None:
    for key, path in paths.items():
        if key.startswith("_"):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows) -> None:
    """Append-only writer for raw generation output.

    Spec section 44 Rule 6: raw Phase 1 outputs are experimental evidence and
    must never be overwritten.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Proof rendering
#
# ProofWriter's `proofsWithIntermediates[].representation` is a small
# S-expression grammar over theory-level symbols:
#
#     tripleN  -> a given fact          (`triples["tripleN"]["text"]`)
#     ruleN    -> a given rule          (`rules["ruleN"]["text"]`)
#     intM     -> an intermediate       (question-level `intermediates["intM"]`)
#     (a b -> (ruleN % intM))           infer intM from premises a, b via ruleN
#
# Examples seen in the data:
#     (triple11)
#     ((triple2) -> (rule6 % int2))
#     ((((triple2) -> (rule6 % int2)) triple3) -> (rule7 % int1))
#
# The renderer walks the tree post-order and emits one natural-language step
# per inference, so the demonstration reasoning is the dataset's *verified*
# proof rather than anything generated by a model (spec section 35).
# ---------------------------------------------------------------------------

class ProofError(Exception):
    """Raised when a proof representation cannot be parsed or resolved."""


_TOKEN_RE = re.compile(r"\s*(->|[()%]|[A-Za-z_][A-Za-z0-9_]*)")


def _tokenize(text: str) -> list:
    tokens = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match:
            if text[pos:].strip() == "":
                break
            raise ProofError(f"unexpected character at {pos}: {text[pos:pos + 20]!r}")
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


def _parse_expr(tokens: list, i: int):
    """Returns (node, next_index). node is a tuple:
       ('atom', name) | ('pct', lhs, rhs) | ('group', items, conclusion|None)
    """
    if i >= len(tokens):
        raise ProofError("unexpected end of proof")
    token = tokens[i]

    if token == "(":
        i += 1
        items = []
        while i < len(tokens) and tokens[i] not in ("->", ")"):
            node, i = _parse_expr(tokens, i)
            if i < len(tokens) and tokens[i] == "%":
                i += 1
                rhs, i = _parse_expr(tokens, i)
                node = ("pct", node, rhs)
            items.append(node)
        conclusion = None
        if i < len(tokens) and tokens[i] == "->":
            i += 1
            conclusion, i = _parse_expr(tokens, i)
            if i < len(tokens) and tokens[i] == "%":
                i += 1
                rhs, i = _parse_expr(tokens, i)
                conclusion = ("pct", conclusion, rhs)
        if i >= len(tokens) or tokens[i] != ")":
            raise ProofError("unbalanced parentheses in proof")
        i += 1
        # `(ruleN % intM)` is a labelled conclusion. Return it as the pct node
        # itself rather than wrapping it in a redundant single-item group.
        if conclusion is None and len(items) == 1 and items[0][0] == "pct":
            return items[0], i
        return ("group", items, conclusion), i

    if token in (")", "->", "%"):
        raise ProofError(f"unexpected token {token!r}")

    return ("atom", token), i + 1


def _atom_key(name: str) -> tuple:
    """'triple12' -> ('triple', 'triple12'); 'rule3' -> ('rule', 'rule3').

    'intN' and 'nafN' both resolve against the intermediates table. `nafN` is a
    negation-as-failure premise ("X is not Y"), which ProofWriter stores
    alongside ordinary intermediate conclusions rather than in its own table.
    """
    lowered = name.lower()
    if lowered.startswith("triple"):
        return "triple", lowered
    if lowered.startswith("rule"):
        return "rule", lowered
    if lowered.startswith("int") or lowered.startswith("naf"):
        return "int", lowered
    raise ProofError(f"unrecognised proof symbol {name!r}")


def _unwrap(node: tuple) -> tuple:
    """Collapse a single-item, arrow-less group down to its only child.

    `_parse_expr` wraps every parenthesised form in a group, so a conclusion
    written as `(rule6 % int2)` arrives as ('group', [pct], None). Callers that
    care about the rule label need the pct itself.
    """
    while node[0] == "group" and node[2] is None and len(node[1]) == 1:
        node = node[1][0]
    return node


def _sentence(text: str) -> str:
    """Normalise a proof fragment so it can be joined into a sentence."""
    return re.sub(r"[\s.;]+$", "", (text or "").strip())


class ProofRenderer:
    """Renders one verified proof tree into natural-language reasoning steps."""

    def __init__(self, theory_row: dict, intermediates: dict):
        self.triples = theory_row.get("triples") or {}
        self.rules = theory_row.get("rules") or {}
        self.intermediates = intermediates or {}

    def _lookup(self, bucket: str, key: str) -> str:
        table = {
            "triple": self.triples,
            "rule": self.rules,
            "int": self.intermediates,
        }[bucket]
        if key not in table:
            raise ProofError(f"missing {bucket} {key!r} in theory")
        entry = table[key]
        text = entry.get("text") if isinstance(entry, dict) else entry
        if not text:
            raise ProofError(f"empty text for {bucket} {key!r}")
        return text.strip()

    def _conclusion(self, node: tuple, steps: list) -> tuple:
        """Resolve an inference conclusion into (rule_text|None, result_text)."""
        node = _unwrap(node)
        if node[0] == "pct":
            rule_node, result_node = node[1], node[2]
            rule_text = None
            rule_atom = _unwrap(rule_node)
            if rule_atom[0] == "atom":
                bucket, key = _atom_key(rule_atom[1])
                if bucket == "rule":
                    rule_text = self._lookup("rule", key)
            return rule_text, self._text(result_node, steps)
        return None, self._text(node, steps)

    def _text(self, node: tuple, steps: list) -> str:
        """Return the natural-language text of a node, appending any steps."""
        kind = node[0]

        if kind == "atom":
            bucket, key = _atom_key(node[1])
            return self._lookup(bucket, key)

        if kind == "pct":
            return self._text(node[2], steps)

        # group
        _, items, conclusion = node
        if conclusion is None:
            parts = [self._text(item, steps) for item in items]
            if len(parts) == 1:
                return parts[0]
            return " and ".join(_sentence(p) for p in parts)

        premise_texts = []
        for item in items:
            premise_texts.append(self._text(item, steps))

        rule_text, result_text = self._conclusion(conclusion, steps)

        conclusion_str = _sentence(result_text)
        if premise_texts:
            premise_str = " and ".join(f'"{_sentence(p)}"' for p in premise_texts)
            if rule_text:
                steps.append(
                    f'From {premise_str} and the rule "{_sentence(rule_text)}", '
                    f"we can conclude {conclusion_str}."
                )
            else:
                steps.append(f"From {premise_str}, we can conclude {conclusion_str}.")
        elif rule_text:
            steps.append(
                f'By the rule "{_sentence(rule_text)}", we can conclude {conclusion_str}.'
            )
        else:
            steps.append(f"We can conclude {conclusion_str}.")

        return result_text

    def render(self, representation: str) -> tuple:
        """Returns (steps, conclusion_text). Raises ProofError on failure."""
        tokens = _tokenize(representation)
        node, index = _parse_expr(tokens, 0)
        if index != len(tokens):
            raise ProofError(f"trailing tokens in proof: {tokens[index:]}")
        steps: list = []
        conclusion = self._text(node, steps)
        return steps, conclusion


def _normalise(text: str) -> str:
    return re.sub(r"[\s.]+", " ", text or "").strip().lower()


def render_gold_proof(theory_row: dict, question_record: dict) -> tuple:
    """Render a question's verified ProofWriter proof into natural language.

    Returns (reasoning_text, meta). `reasoning_text` is None when no proof could
    be rendered, in which case `meta['error']` explains why.

    Handles the two ProofWriter strategies (spec section 6):
      * 'proof'     — the proof derives the question's positive claim.
      * 'inv-proof' — the proof derives the positive form of a claim that the
                      question states in negated form; this is how a False
                      answer is established under the closed world assumption.
                      The renderer states that pivot explicitly.
    """
    alternatives = question_record.get("proofsWithIntermediates") or []
    if not alternatives:
        return None, {"error": "no proofsWithIntermediates"}

    chosen = alternatives[0]
    representation = chosen.get("representation")
    if not representation:
        return None, {"error": "empty representation"}

    renderer = ProofRenderer(theory_row, chosen.get("intermediates"))
    try:
        steps, conclusion = renderer.render(representation)
    except ProofError as exc:
        return None, {"error": f"ProofError: {exc}"}

    question_text = question_record.get("question", "").strip()
    answer = question_record.get("answer")
    strategy = question_record.get("strategy")

    lines = list(steps)
    if not lines:
        return None, {"error": "proof produced no inference steps"}

    question_clean = _sentence(question_text)
    # Validate: the rendered conclusion must agree with the question.
    if bool(answer) is True:
        if _normalise(conclusion) != _normalise(question_text):
            return None, {
                "error": "conclusion does not match question",
                "conclusion": conclusion,
                "question": question_text,
            }
        lines.append(f"Therefore, {question_clean}.")
    else:
        lines.append(
            f'The question asks whether "{question_clean}". The facts and rules '
            f'entail that {_sentence(conclusion)}, so this claim is false.'
        )

    meta = {
        "strategy": strategy,
        "qdep": question_record.get("QDep"),
        "proof_index": 0,
        "num_alternatives": len(alternatives),
        "num_steps": len(steps),
        "conclusion": conclusion,
    }
    return "\n".join(lines), meta


# ---------------------------------------------------------------------------
# Dataset loading and pool selection
# ---------------------------------------------------------------------------

def load_theories(cfg: dict) -> list:
    """Load every theory row from the configured dataset split."""
    from datasets import load_dataset

    ds_cfg = cfg["dataset"]
    log.info(
        "Loading %s [%s] split=%s",
        ds_cfg["name"], ds_cfg["configuration"], ds_cfg["split"],
    )
    ds = load_dataset(ds_cfg["name"], ds_cfg["configuration"], split=ds_cfg["split"])
    rows = [dict(row) for row in ds]
    rows.sort(key=lambda r: r["id"])  # deterministic ordering
    log.info("Loaded %d theories", len(rows))
    return rows


def question_id(theory_id: str, question_key: str) -> str:
    """Stable, human-traceable ID. Spec section 43 requires the ID chain from
    final result back to the original dataset example to be walkable."""
    return f"{theory_id}::{question_key}"


def build_candidates(theories: list, cfg: dict) -> tuple:
    """Render every eligible question's verified proof, keeping only those that
    render cleanly.

    The pool and the demonstrations are both drawn from this list, so a target
    always has a usable gold proof and a demonstration always has verified
    dataset reasoning (spec sections 7, 35).
    """
    min_qdep = cfg["dataset"]["min_qdep"]
    candidates = []
    skipped = {"shallow": 0, "no_proof": 0, "unrenderable": 0}

    for row in theories:
        proofable = 0
        for question_key, record in (row.get("questions") or {}).items():
            if record.get("QDep", 0) < min_qdep:
                skipped["shallow"] += 1
                continue
            if not record.get("proofsWithIntermediates"):
                skipped["no_proof"] += 1
                continue
            proofable += 1
            reasoning, meta = render_gold_proof(row, record)
            if reasoning is None:
                skipped["unrenderable"] += 1
                continue
            candidates.append({
                "theory_id": row["id"],
                "question_key": question_key,
                "id": question_id(row["id"], question_key),
                "context": row["theory"],
                "question": record["question"],
                "gold_answer": "True" if bool(record["answer"]) else "False",
                "gold_proof": reasoning,
                "reasoning_depth": record.get("QDep"),
                "strategy": record.get("strategy"),
                "render_meta": meta,
            })

    candidates.sort(key=lambda c: c["id"])
    log.info(
        "Candidates: %d renderable (skipped %d shallow, %d without proofs, %d unrenderable)",
        len(candidates), skipped["shallow"], skipped["no_proof"], skipped["unrenderable"],
    )
    return candidates, skipped


def select_demonstrations(candidates: list, cfg: dict) -> tuple:
    """Choose the two fixed demonstrations (spec sections 7 and 8).

    One demonstration has gold answer True and one has gold answer False, both
    with QDep >= min_qdep so the reasoning is non-trivial. They come from two
    different theories, and both theories are excluded from the target pool so
    no demonstration can leak information about a target.
    """
    demos = {}
    used_theories = set()
    for want_answer in (True, False):
        want = "True" if want_answer else "False"
        for cand in candidates:  # sorted by id -> deterministic
            if cand["theory_id"] in used_theories:
                continue
            if cand["gold_answer"] != want:
                continue
            demos[want_answer] = cand
            used_theories.add(cand["theory_id"])
            break

    if True not in demos or False not in demos:
        missing = [k for k in (True, False) if k not in demos]
        raise RuntimeError(
            f"Could not build both demonstrations; missing gold answer(s) {missing}."
        )
    return demos[True], demos[False]


def select_pool(candidates: list, cfg: dict, demo_theories: set) -> list:
    """Select the 500 balanced targets before any inference (spec section 51).

    Deterministic: candidates are gathered in sorted order and shuffled with the
    configured seed, so the same config always yields the same pool.
    """
    ds_cfg = cfg["dataset"]
    seed = cfg["generation"]["seed"]

    by_answer = {"True": [], "False": []}
    for cand in candidates:
        if cand["theory_id"] in demo_theories:
            continue
        by_answer[cand["gold_answer"]].append(cand)

    rng = random.Random(seed)
    for bucket in by_answer.values():
        rng.shuffle(bucket)

    need = {"True": ds_cfg["pool_true"], "False": ds_cfg["pool_false"]}
    for answer, count in need.items():
        if len(by_answer[answer]) < count:
            raise RuntimeError(
                f"Only {len(by_answer[answer])} eligible questions with gold answer "
                f"{answer}; need {count}. Widen the theory pool or lower min_qdep."
            )

    selected = []
    for answer in ("True", "False"):
        selected.extend(by_answer[answer][: need[answer]])
    selected.sort(key=lambda c: c["id"])

    n_true = sum(1 for c in selected if c["gold_answer"] == "True")
    log.info(
        "Selected %d targets (%d True / %d False) from %d candidates",
        len(selected), n_true, len(selected) - n_true, len(candidates),
    )
    return selected


# ---------------------------------------------------------------------------
# Prompt construction
#
# Spec section 10: Teacher and Student must use the SAME chat-template
# mechanism; their only experimental difference is the presence of the two
# demonstrations. To keep the shared portion byte-identical (which also keeps
# Phase 2's LCS alignment cleaner), the target block is built by one function
# and reused verbatim in both prompts.
# Spec section 4: the target's gold answer and gold proof never appear here.
# ---------------------------------------------------------------------------

def target_block(context: str, question: str) -> str:
    return (
        f"Context:\n{context.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Reasoning:"
    )


def demonstration_block(index: int, demo: dict) -> str:
    return (
        f"Demonstration {index}\n\n"
        f"Context:\n{demo['context'].strip()}\n\n"
        f"Question:\n{demo['question'].strip()}\n\n"
        f"Reasoning:\n{demo['gold_proof'].strip()}\n\n"
        f"Final Answer: {demo['gold_answer']}"
    )


def build_student_prompt(record: dict, cfg: dict) -> str:
    """Zero-shot Student input: target context + target question. No demos."""
    return target_block(record["context"], record["question"])


def build_teacher_prompt(record: dict, demos: list, cfg: dict) -> str:
    """Two-shot Teacher input: 2 demonstrations + target context + question."""
    blocks = [demonstration_block(i + 1, d) for i, d in enumerate(demos)]
    return "\n\n".join(blocks) + "\n\nTarget\n\n" + target_block(
        record["context"], record["question"]
    )


# ---------------------------------------------------------------------------
# Answer parsing
#
# Spec section 26: binary mapping only. No generic multiple-choice index map,
# no fallback index, no silent conversion.
# Spec section 9: a response with no valid final answer becomes INVALID and
# counts as incorrect. Do NOT rerun the model merely because parsing failed.
# ---------------------------------------------------------------------------

_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*[:\-]\s*(True|False)", re.IGNORECASE)


def parse_answer(text: str, cfg: dict) -> str:
    """Extract the final answer. Returns 'True', 'False', or 'INVALID'."""
    if not text:
        return cfg["parser"]["invalid_label"]
    matches = _FINAL_ANSWER_RE.findall(text)
    if not matches:
        return cfg["parser"]["invalid_label"]
    # Spec section 9: the evaluator extracts only the final answer.
    return matches[-1].capitalize()


def reasoning_only(text: str) -> str:
    """Everything the model produced, verbatim. Retained for analysis and for
    Phase 2 distillation (spec section 9)."""
    return text.strip()


# ---------------------------------------------------------------------------
# Model loading and generation
# ---------------------------------------------------------------------------

def load_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg["model"]
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

    log.info("Loading tokenizer %s@%s", model_cfg["name"], model_cfg["revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg["revision"], trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info("Loading model %s@%s", model_cfg["name"], model_cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg["revision"],
        torch_dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def render_chat(tokenizer, user_content: str, cfg: dict) -> str:
    messages = []
    system_message = (cfg["prompt"].get("system_message") or "").strip()
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_batch(model, tokenizer, prompts: list, cfg: dict) -> list:
    import torch

    gen_cfg = cfg["generation"]
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=bool(gen_cfg["do_sample"]),
            temperature=gen_cfg["temperature"] if gen_cfg["do_sample"] else None,
            top_p=gen_cfg["top_p"] if gen_cfg["do_sample"] else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = enc["input_ids"].shape[1]
    return tokenizer.batch_decode(
        output[:, input_length:], skip_special_tokens=True
    )


def run_condition(model, tokenizer, records: list, prompts: list, condition: str,
                  cfg: dict, batch_size: int, raw_path: Path) -> None:
    """Generate for one condition, appending each result to disk immediately so
    a Colab disconnect never loses completed work (spec sections 16, 54)."""
    total = len(records)
    done = 0
    for start in range(0, total, batch_size):
        chunk_records = records[start:start + batch_size]
        chunk_prompts = prompts[start:start + batch_size]
        started = time.time()
        outputs = generate_batch(model, tokenizer, chunk_prompts, cfg)
        rows = []
        for record, prompt, output in zip(chunk_records, chunk_prompts, outputs):
            rows.append({
                "id": record["id"],
                "theory_id": record["theory_id"],
                "condition": condition,
                "prompt": prompt,
                "output": output,
                "prompt_version": cfg["prompt"]["template_version"],
                "parser_version": cfg["parser"]["version"],
                "model_name": cfg["model"]["name"],
                "model_revision": cfg["model"]["revision"],
                "generation_parameters": {
                    "do_sample": cfg["generation"]["do_sample"],
                    "temperature": cfg["generation"]["temperature"],
                    "top_p": cfg["generation"]["top_p"],
                    "max_new_tokens": cfg["generation"]["max_new_tokens"],
                    "seed": cfg["generation"]["seed"],
                },
                "demonstration_ids": (
                    [d["id"] for d in record.get("_demos", [])] if condition == "few_shot" else []
                ),
                "timestamp": now_iso(),
            })
        append_jsonl(raw_path, rows)
        done += len(rows)
        elapsed = time.time() - started
        log.info(
            "[%s] %d/%d generated (%.1fs for last %d)",
            condition, done, total, elapsed, len(rows),
        )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(zero_shot_correct: bool, few_shot_correct: bool) -> str:
    """Spec section 13. Every target gets exactly one category."""
    if not zero_shot_correct and few_shot_correct:
        return "golden"
    if not zero_shot_correct and not few_shot_correct:
        return "both_failed"
    if zero_shot_correct and few_shot_correct:
        return "both_passed"
    return "regression"  # zero-shot correct, few-shot wrong (Category D)


CATEGORY_FILES = {
    "golden": "golden_records",
    "both_failed": "both_failed",
    "both_passed": "both_passed",
    "regression": "regressions",
}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_prepare(cfg: dict, paths: dict, limit: int | None) -> None:
    theories = load_theories(cfg)
    candidates, skipped = build_candidates(theories, cfg)

    demo_true, demo_false = select_demonstrations(candidates, cfg)
    demos = [demo_true, demo_false]
    demo_theories = {demo_true["theory_id"], demo_false["theory_id"]}
    log.info(
        "Demonstrations: %s (%s) and %s (%s)",
        demo_true["id"], demo_true["gold_answer"],
        demo_false["id"], demo_false["gold_answer"],
    )

    pool = select_pool(candidates, cfg, demo_theories)
    if limit:
        pool = pool[:limit]

    write_jsonl(paths["pool"], pool)
    write_json(paths["demonstrations"], {
        "demonstrations": demos,
        "prompt_version": cfg["prompt"]["template_version"],
        "selection_rule": "fixed_pair_theory_excluded",
        "excluded_theories": sorted(demo_theories),
        "created": now_iso(),
    })

    counts = {}
    for record in pool:
        counts[record["gold_answer"]] = counts.get(record["gold_answer"], 0) + 1
    write_json(paths["phase1_config"], {
        "phase": 1,
        "stage": "prepare",
        "created": now_iso(),
        "artifact_root": str(paths["_root"]),
        "config": cfg,
        "candidates_total": len(candidates),
        "candidates_skipped": skipped,
        "pool_size": len(pool),
        "pool_answer_counts": counts,
        "demonstration_ids": [d["id"] for d in demos],
        "excluded_theories": sorted(demo_theories),
    })
    log.info("Wrote pool (%d) and demonstrations to %s", len(pool), paths["_root"])


def stage_generate(cfg: dict, paths: dict, batch_size: int, limit: int | None,
                   conditions: list) -> None:
    pool = read_jsonl(paths["pool"])
    if not pool:
        sys.exit(f"No pool found at {paths['pool']}. Run --stage prepare first.")
    if limit:
        pool = pool[:limit]

    demos_payload = json.loads(paths["demonstrations"].read_text(encoding="utf-8"))
    demos = demos_payload["demonstrations"]

    # Work out what is missing BEFORE loading the model, so a resumed run that
    # has nothing left to do costs nothing (spec section 55: resumable, and
    # section 44 Rule 9: never regenerate what already exists).
    plan = {}
    for condition in conditions:
        raw_path = paths["raw_zero_shot"] if condition == "zero_shot" else paths["raw_few_shot"]
        existing = {row["id"] for row in read_jsonl(raw_path)}
        todo = [record for record in pool if record["id"] not in existing]
        plan[condition] = (raw_path, todo)
        log.info(
            "[%s] %d/%d already generated, %d to do",
            condition, len(pool) - len(todo), len(pool), len(todo),
        )

    if all(not todo for _, todo in plan.values()):
        log.info("Nothing to generate; skipping model load.")
        return

    model, tokenizer = load_model(cfg)

    for condition in conditions:
        raw_path, todo = plan[condition]
        if not todo:
            continue

        records, prompts = [], []
        for record in todo:
            if condition == "zero_shot":
                prompt = build_student_prompt(record, cfg)
                record["_demos"] = []
            else:
                prompt = build_teacher_prompt(record, demos, cfg)
                record["_demos"] = demos
            records.append(record)
            prompts.append(render_chat(tokenizer, prompt, cfg))

        log.info("[%s] generating %d records", condition, len(records))
        run_condition(model, tokenizer, records, prompts, condition,
                      cfg, batch_size, raw_path)


def stage_classify(cfg: dict, paths: dict) -> None:
    pool = read_jsonl(paths["pool"])
    if not pool:
        sys.exit(f"No pool found at {paths['pool']}. Run --stage prepare first.")

    zero_shot = {row["id"]: row for row in read_jsonl(paths["raw_zero_shot"])}
    few_shot = {row["id"]: row for row in read_jsonl(paths["raw_few_shot"])}

    demos_payload = json.loads(paths["demonstrations"].read_text(encoding="utf-8"))
    demo_ids = [d["id"] for d in demos_payload["demonstrations"]]
    dataset_split = cfg["dataset"]["split"]

    records = []
    for target in pool:
        zs = zero_shot.get(target["id"])
        fs = few_shot.get(target["id"])
        if zs is None or fs is None:
            log.warning("Skipping %s: missing %s generation",
                        target["id"],
                        "zero-shot" if zs is None else "few-shot")
            continue

        zs_answer = parse_answer(zs["output"], cfg)
        fs_answer = parse_answer(fs["output"], cfg)
        # Spec section 44 Rule 4: classification compares against the gold
        # dataset label, never against generated text alone.
        zs_correct = zs_answer == target["gold_answer"]
        fs_correct = fs_answer == target["gold_answer"]

        records.append({
            "id": target["id"],
            "theory_id": target["theory_id"],
            "question_key": target["question_key"],
            "context": target["context"],
            "question": target["question"],
            "gold_answer": target["gold_answer"],
            "gold_proof": target["gold_proof"],
            "reasoning_depth": target["reasoning_depth"],

            "zero_shot_reasoning": reasoning_only(zs["output"]),
            "zero_shot_answer": zs_answer,
            "zero_shot_correct": zs_correct,

            "few_shot_reasoning": reasoning_only(fs["output"]),
            "few_shot_answer": fs_answer,
            "few_shot_correct": fs_correct,

            "category": classify(zs_correct, fs_correct),

            "prompt_version": cfg["prompt"]["template_version"],
            "parser_version": cfg["parser"]["version"],
            "model_name": cfg["model"]["name"],
            "model_revision": cfg["model"]["revision"],
            "generation_parameters": zs["generation_parameters"],
            "demonstration_ids": demo_ids,
            "dataset_split": dataset_split,
            "dataset_configuration": cfg["dataset"]["configuration"],
            "timestamp": now_iso(),
        })

    records.sort(key=lambda r: r["id"])
    write_jsonl(paths["all_records"], records)  # authoritative (spec section 41)

    for category, key in CATEGORY_FILES.items():
        subset = [r for r in records if r["category"] == category]
        write_jsonl(paths[key], subset)
        log.info("%-12s %d", category, len(subset))

    by_depth = {}
    for record in records:
        by_depth[record["reasoning_depth"]] = by_depth.get(record["reasoning_depth"], 0) + 1

    counts = {c: sum(1 for r in records if r["category"] == c) for c in CATEGORY_FILES}
    golden = [r for r in records if r["category"] == "golden"]

    # Preserve the prepare-stage metadata rather than overwriting it: spec
    # section 39 requires every run's configuration to stay recoverable, and the
    # pool-selection record is part of that evidence.
    prepare_meta = None
    if paths["phase1_config"].exists():
        try:
            existing_meta = json.loads(paths["phase1_config"].read_text(encoding="utf-8"))
            if existing_meta.get("stage") == "prepare":
                prepare_meta = existing_meta
        except (json.JSONDecodeError, OSError):
            prepare_meta = None

    summary = {
        "phase": 1,
        "stage": "classify",
        "created": now_iso(),
        "artifact_root": str(paths["_root"]),
        "total_records": len(records),
        "category_counts": counts,
        "pool_answer_counts": {
            "True": sum(1 for r in records if r["gold_answer"] == "True"),
            "False": sum(1 for r in records if r["gold_answer"] == "False"),
        },
        "golden_answer_counts": {
            "True": sum(1 for r in golden if r["gold_answer"] == "True"),
            "False": sum(1 for r in golden if r["gold_answer"] == "False"),
        },
        "reasoning_depth_counts": dict(sorted(by_depth.items(), key=lambda kv: str(kv[0]))),
        "invalid_counts": {
            "zero_shot": sum(1 for r in records if r["zero_shot_answer"] == "INVALID"),
            "few_shot": sum(1 for r in records if r["few_shot_answer"] == "INVALID"),
        },
        "config": cfg,
        "prepare": prepare_meta,
    }
    write_json(paths["phase1_config"], summary)
    log.info("Phase 1 complete: %d records, %d Golden", len(records), counts["golden"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 — Golden Record discovery for Attention Calibration Distillation 02"
    )
    parser.add_argument("--stage", default="all",
                        choices=["prepare", "generate", "classify", "all"])
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="generation batch size (default 1: sequences vary a lot in length)")
    parser.add_argument("--conditions", default="zero_shot,few_shot",
                        help="comma-separated subset of zero_shot,few_shot for --stage generate")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N theories/targets (smoke testing)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    paths = artifact_paths(cfg)
    ensure_dirs(paths)
    log.info("Artifact root: %s", paths["_root"])

    if args.stage in ("prepare", "all"):
        stage_prepare(cfg, paths, args.limit)
    if args.stage in ("generate", "all"):
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = set(conditions) - {"zero_shot", "few_shot"}
        if unknown:
            sys.exit(f"Unknown condition(s): {sorted(unknown)}")
        stage_generate(cfg, paths, args.batch_size, args.limit, conditions)
    if args.stage in ("classify", "all"):
        stage_classify(cfg, paths)


if __name__ == "__main__":
    main()
