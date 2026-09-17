# Attention Calibration Distillation 02

Testing whether attention calibration + hidden-state distillation can train Qwen2.5-3B-Instruct to solve, zero-shot, ProofWriter instances that it fails zero-shot but solves when given two verified solved demonstrations.

The experimental contract is `Attention Calibration Distillation — Three-Phase Experimental Project Specification.md`. `HyperICL.pdf` is the architectural source of truth. Where the two disagree, the spec governs the *experiment* and HyperICL governs the *mechanism*.

## The three phases

| Phase | File | Question it answers |
|---|---|---|
| 1 — Discovery | `phase1.py` | Which instances does the model fail zero-shot but solve with two verified demonstrations? |
| 2 — Training | `phase2.py` *(not yet written)* | Can HyperICL attention calibration + distillation transfer that behaviour into the zero-shot Student? |
| 3 — Evaluation | `phase3.py` *(not yet written)* | Does the calibrated model recover held-out Golden Records, and at what regression cost? |

There is exactly one executable Python file per phase, per spec §55.

## Experimental conditions

Both conditions use the **same model** and the **same Qwen chat template**. Their only difference is the demonstration block.

```
Teacher (2-shot)                          Student (0-shot)
  Demonstration 1                           Context: <target>
  Demonstration 2                           Question: <target question>
  Target                                    Reasoning:
  Context: <target>
  Question: <target question>
  Reasoning:
```

The target's gold answer and gold proof are **never** placed in either prompt. They exist only in the dataset and the evaluator.

A **Golden Record** is strictly:

```
gold_answer == few_shot_answer  AND  gold_answer != zero_shot_answer
```

Everything else is retained too — `both_failed`, `both_passed`, and `regressions` — because all four categories are informative (spec §13, §44 Rule 7).

## Quick start

```bash
pip install -r requirements.txt

# Build the 500-target pool and the two demonstrations (CPU only, no model needed)
python phase1.py --stage prepare

# Full Phase 1 run (needs a GPU)
python phase1.py --stage all
```

### In Colab

Colab's local filesystem is temporary, so point the artifact root at mounted Drive. The same commands then work unchanged:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
export ACD_ROOT=/content/drive/MyDrive/acd02
git clone <this-repo> && cd <this-repo>
python phase1.py --stage all
```

`ACD_ROOT` is the only thing that changes between local and Colab runs.

### Resuming

Every stage is resumable. Raw generations are **append-only** — spec §44 Rule 6 makes them experimental evidence, so the file is never truncated and completed work is never regenerated (§44 Rule 9). A resumed run with nothing left to do skips model loading entirely.

```bash
python phase1.py --stage generate           # generates only what is missing
python phase1.py --stage classify           # derive categories from cached raw output
```

Other flags: `--limit N` (smoke test on the first N targets), `--batch-size`, `--conditions zero_shot`, `--verbose`.

## Phase 1 artifacts

Spec §41. `all_records.jsonl` is authoritative; every other file is derived from it and can be rebuilt at any time without touching the model.

```
data/
  phase1/pool.jsonl                 the 500 selected targets (written BEFORE inference)
  demonstrations/demonstrations.json the two fixed verified demonstrations
results/phase1/
  raw/zero_shot.jsonl               append-only raw generations + prompts
  raw/few_shot.jsonl                append-only raw generations + prompts
  all_records.jsonl                 AUTHORITATIVE — one record per target
  golden_records.jsonl              zero-shot wrong  / few-shot correct
  both_failed.jsonl                 zero-shot wrong  / few-shot wrong
  both_passed.jsonl                 zero-shot correct / few-shot correct
  regressions.jsonl                 zero-shot correct / few-shot wrong
  phase1_config.json                pool selection + full config + run summary
```

Every record carries its original dataset ID, so a final result can be traced back through Phase 1 to the original ProofWriter theory and gold proof (spec §43).

## Design notes

**Dataset.** `arqa39/proofwriter-source`, configuration `CWA-depth-2`, split `train`. Closed World Assumption keeps the answer space to exactly `{True, False}` with no `Unknown`, which is what makes the binary label mapping in spec §26 safe. Depth-2 is the shallowest configuration that still contains real multi-hop reasoning.

**Target eligibility.** A question is eligible only if it has `QDep >= 1` (at least one inference step, so it is not a direct fact lookup) *and* the dataset ships a verified proof for it. Of 33,708 shallow questions and 10,836 without proofs, 25,616 candidates remain.

**Demonstrations.** Two fixed examples — one gold `True`, one gold `False` — from two theories that are then excluded from the target pool, so no demonstration can leak information about a target (spec §8). They use the dataset's verified proofs, never model-generated reasoning (spec §35).

**Proof rendering.** ProofWriter proofs are small S-expressions over `tripleN` (facts), `ruleN` (rules) and `intN`/`nafN` (intermediate conclusions, including negation-as-failure premises). `phase1.py` parses the tree and emits one natural-language step per inference, then validates that the rendered conclusion actually matches the question. A proof that fails to render or validate is rejected rather than approximated.

**False answers.** ProofWriter proves a `False` answer by refuting it — the gold proof derives the *positive* claim instead. The renderer states that pivot explicitly, so a demonstration never appears to prove the opposite of its own answer.

**Answer mapping.** Binary only. No generic multiple-choice index mapping, no fallback index, no silent conversion (spec §26). A response with no parsable final answer becomes `INVALID` and counts as incorrect, and the model is **not** rerun (spec §9).

## Rules this code enforces

- Teacher and Student are byte-identical over the target block; the Student prompt is a verbatim suffix of the Teacher prompt. Verified across all 500 targets.
- No gold target information reaches either prompt.
- The Student never receives demonstrations.
- Raw Phase 1 outputs are never overwritten.
- Every category is retained, including failures.
- IDs are preserved through every transformation.
- The test set is never modified after training begins.
