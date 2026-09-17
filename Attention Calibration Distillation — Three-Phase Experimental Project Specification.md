# Attention Calibration Distillation for Demonstration-Free Text Reasoning

## 1. Project Status

This project is a continuation and redesign of an earlier research/engineering attempt called:

**Attention Calibration Distillation for Demonstration-Free Text Reasoning**

The previous implementation was inspired by Hyper-ICL and attempted to transfer reasoning behavior from a few-shot Teacher to a zero-shot Student through attention calibration and hidden-state distillation.

The previous implementation encountered multiple engineering and dataset-quality problems, including:

- Out-of-memory failures when Teacher and Student were run simultaneously.
- Teacher/Student sequence-length mismatch.
- Incorrect target-label mapping.
- Dataset contamination caused by inconsistent prompting.
- Unreliable evaluation on BBH.
- Insufficient trustworthy Golden Records.
- An earlier reported final result of 5/11 (45%) was subsequently identified as historically unreliable. The corrected historical result to use is **1/11**.

The old report documents the original architecture and engineering history. It should be treated as historical context, not as proof that the previous experiment succeeded.

This new project is intentionally narrower and cleaner.

The goal is to determine whether the same small language model can be trained to reproduce useful behavior that it demonstrates when given a small number of verified reasoning examples.

---

# 2. Research Objective

The central question is:

> **If Qwen2.5-3B-Instruct can solve an instance when given two solved reasoning demonstrations, but fails on the same instance with no demonstrations, can attention calibration + distillation train the model to solve that instance zero-shot?**

The project does **not** assume that the model contains inaccessible "knowledge" or that the Teacher is revealing hidden knowledge.

The experimentally defensible statement is simpler:

> The model can solve some instances under a few-shot condition while failing on those same instances under a zero-shot condition. We investigate whether attention calibration and distillation can transfer useful behavior from the few-shot condition into the zero-shot condition.

---

# 3. Core Experimental Principle

The experiment uses the **same model** as Teacher and Student:

**Qwen2.5-3B-Instruct**

There is no requirement for a larger Teacher.

The Teacher and Student differ only in their input conditions:

### Teacher

Receives:

- Two solved demonstrations.
- Each demonstration contains:
  - Context
  - Question
  - Verified reasoning/proof
  - Verified answer
- Then receives the target:
  - Context
  - Question
- The Teacher must generate the target reasoning and answer.

### Student

Receives:

- Target context.
- Target question.
- No demonstrations.
- No target reasoning.
- No target answer.

The Student must generate the target reasoning and answer.

This distinction is fundamental.

---

# 4. CRITICAL PROMPTING RULE

## NEVER give the target answer or target reasoning to either model.

The target's gold answer and gold proof exist only in the dataset/evaluation system.

They are never inserted into the Teacher or Student target prompt.

Correct:

```text
Demonstration 1:
Context: ...
Question: ...
Reasoning: [verified demonstration proof]
Final Answer: True

Demonstration 2:
Context: ...
Question: ...
Reasoning: [verified demonstration proof]
Final Answer: False

Target:
Context: ...
Question: ...

[MODEL GENERATES REASONING AND ANSWER]
```

Incorrect:

```text
Target:
Context: ...
Question: ...
Reasoning: [gold reasoning]
Answer: [gold answer]
```

The second version invalidates the experiment.

---

# 5. Dataset

The project will use **ProofWriter**, specifically a configuration appropriate for deterministic binary reasoning.

The preferred initial configuration is:

**ProofWriter Closed World Assumption (CWA)**

This keeps the final answer space simple:

```text
True
False
```

ProofWriter is appropriate because the reasoning problem is self-contained:

- Natural-language facts.
- Natural-language rules.
- A question.
- A deterministic answer.
- Explicit proof/reasoning information.

This avoids the external-world-knowledge confound that made StrategyQA unsuitable for this experiment.

---

# 6. Why ProofWriter

The dataset provides the exact information required for this experiment:

```text
Context / Facts
Rules
Question
Gold Answer
Gold Proof
Reasoning Depth
```

A proof can contain intermediate conclusions, for example:

```text
Fact:
The cow is big.

Rule:
If something is big then it chases the dog.

Intermediate conclusion:
The cow chases the dog.

Rule:
If the cow chases the dog then the cow sees the rabbit.

Final conclusion:
The cow sees the rabbit.

Answer:
True
```

The gold proof is therefore available for constructing **verified demonstrations** without asking Qwen to generate the demonstrations.

---

# 7. Demonstration Construction

The two demonstrations are selected from ProofWriter and use the dataset's verified proof/reasoning.

They are prepared once and cached.

For every Teacher target, the same two demonstration examples may be reused unless the implementation explicitly establishes a different deterministic demonstration-selection strategy.

A demonstration contains:

```text
Context
Question
Verified Reasoning / Proof
Verified Answer
```

Example structure:

```text
Demonstration 1

Context:
[ProofWriter context and rules]

Question:
[ProofWriter question]

Reasoning:
[verified ProofWriter proof]

Final Answer:
True
```

```text
Demonstration 2

Context:
[ProofWriter context and rules]

Question:
[ProofWriter question]

Reasoning:
[verified ProofWriter proof]

Final Answer:
False
```

Then the target begins.

```text
Target

Context:
[target context and rules]

Question:
[target question]

Reasoning:
```

The model generates the remainder.

---

# 8. Demonstration Restrictions

Demonstrations must not leak information about the target.

At minimum:

1. Demonstrations must be separate examples from the target.
2. Preferably, demonstrations should come from different ProofWriter theories than the target.
3. Demonstration answers must come from the dataset's verified labels.
4. Demonstration reasoning must come from verified dataset proofs.
5. Do not ask Qwen to solve demonstrations during Phase 1.
6. Do not filter demonstrations by whether Qwen can solve them.
7. Do not use target-specific information in demonstrations.
8. Do not include the target's gold proof anywhere in the Teacher prompt.

The demonstrations are **fixed solved examples**, not examples selected because the model already knows how to solve them.

---

# 9. Output Format

Both Teacher and Student should be instructed to produce:

```text
Reasoning:
...

Final Answer: True
```

or:

```text
Reasoning:
...

Final Answer: False
```

The evaluator must extract only the final answer.

The reasoning is retained for analysis and later distillation.

If the model fails to produce a valid final answer:

```text
predicted_answer = INVALID
```

and the prediction counts as incorrect.

Do **not** rerun the model merely because parsing failed.

This preserves deterministic experiment accounting.

---

# 10. Prompt Consistency

Qwen2.5-3B-Instruct must be invoked using its official/instruction-compatible chat formatting consistently.

The previous project suffered from a serious contamination problem because zero-shot evaluation was performed with raw formatting rather than the Qwen chat template. The historical report explicitly identified this as a source of artificial failures.

Therefore:

> **Teacher and Student must both use the same Qwen chat-template mechanism.**

Their only experimental difference should be the presence or absence of the two demonstrations.

---

# 11. Phase 1 — Golden Record Discovery

## Objective

Find instances where:

```text
Zero-shot Student = WRONG
Few-shot Teacher = CORRECT
```

These are the **Golden Records**.

The same target must be evaluated under both conditions.

---

## 11.1 Phase 1 Teacher Condition

Input:

```text
[2 verified solved demonstrations]

Target context
Target question
```

The Teacher generates:

```text
Reasoning
Final Answer
```

The target answer is NOT supplied.

---

## 11.2 Phase 1 Student Condition

Input:

```text
Target context
Target question
```

No demonstrations.

No target reasoning.

No target answer.

The Student generates:

```text
Reasoning
Final Answer
```

---

# 12. Phase 1 Must Record EVERYTHING

Do not save only Golden Records.

Every evaluated target must produce one persistent record.

This is important because the project must never need to rerun Qwen merely to reconstruct what happened.

Each record should contain at least:

```text
id
theory_id
context
question
gold_answer
gold_proof
reasoning_depth

zero_shot_reasoning
zero_shot_answer
zero_shot_correct

few_shot_reasoning
few_shot_answer
few_shot_correct

category
```

Recommended additional metadata:

```text
prompt_version
model_name
model_revision
generation_parameters
demonstration_ids
dataset_split
timestamp
```

---

# 13. Phase 1 Classification

Every target is assigned exactly one category.

### Category A — Golden Record

```text
Zero-shot: WRONG
Few-shot:  CORRECT
```

This is the primary training population.

---

### Category B — Both Fail

```text
Zero-shot: WRONG
Few-shot:  WRONG
```

These must be retained.

They are useful for diagnosing examples that even the two-shot condition cannot solve.

They must NOT be treated as Golden Records.

---

### Category C — Both Pass

```text
Zero-shot: CORRECT
Few-shot:  CORRECT
```

These demonstrate that the model already solves the instance without demonstrations.

They are not Golden Records.

---

### Category D — Regression

```text
Zero-shot: CORRECT
Few-shot:  WRONG
```

These indicate that the demonstrations actually hurt performance on that target.

They must be retained because they are scientifically useful and provide evidence about the behavior of the prompting setup.

---

# 14. Golden Record Definition

A Golden Record is therefore strictly:

```text
gold_answer == few_shot_answer
AND
gold_answer != zero_shot_answer
```

Equivalently:

```text
few_shot_correct = True
zero_shot_correct = False
```

No other record qualifies.

---

# 15. Phase 1 Data Flow

The complete flow is:

```text
ProofWriter
     |
     v
Select target
     |
     +----------------------------+
     |                            |
     v                            v
Zero-shot Student          2-shot Teacher
     |                            |
     |                            |
     v                            v
Student output             Teacher output
     |                            |
     +-------------+--------------+
                   |
                   v
             Parse answers
                   |
                   v
          Compare with gold
                   |
                   v
            Classify record
                   |
       +-----------+-----------+
       |           |           |
    Golden      Both Fail   Both Pass
       |           |           |
       +-----------+-----------+
                   |
                   v
             Save ALL records
```

No later phase should require rerunning Phase 1 generation simply to discover which records belonged to which category.

---

# 16. Phase 1 Compute Strategy

The Teacher and Student are the same model, so they do not need to occupy GPU memory simultaneously.

For each target:

1. Load/run Qwen.
2. Run zero-shot Student.
3. Save output.
4. Run two-shot Teacher.
5. Save output.
6. Parse and classify.
7. Persist the complete record.

Alternatively, Teacher and Student batches can be run separately if memory and implementation make this more efficient.

The important requirement is:

> **Every generation is cached.**

Do not repeatedly regenerate Teacher or Student answers during later analysis.

The two demonstrations should also be prepared once and reused rather than regenerated for every target.

---

# 17. Phase 1 Dataset Split

Before Phase 2 training, Golden Records must be divided into:

```text
Golden Train
Golden Validation
Golden Test
```

The split must happen **before training**.

Do not select the test set after observing training results.

---

# 18. Important Leakage Rule

ProofWriter can contain multiple questions associated with the same underlying theory/context.

Therefore, if multiple questions share the same theory/context, they should not be randomly distributed across train and test.

Instead:

> **Split by theory ID, not merely by question ID, whenever the dataset structure permits this.**

Otherwise the model may see essentially the same logical world during training and testing.

That would make the final improvement difficult to interpret.

---

# 19. Phase 2 — Attention Calibration + Distillation

## Objective

Train a small parameter-efficient modification of Qwen2.5-3B so that:

```text
Zero-shot calibrated Student
```

learns behavior associated with:

```text
Few-shot Teacher
```

using only Golden Records.

The Teacher's advantage comes from the demonstrations.

The Student must continue to receive **zero demonstrations** during training.

---

# 20. Phase 2 Training Data

Training data consists only of Phase 1 Golden Records.

For every Golden Record:

```text
Teacher input:
2 demonstrations + target

Student input:
target only
```

Teacher target-side outputs:

```text
teacher reasoning
teacher final answer
```

Student target-side outputs:

```text
student reasoning
student final answer
```

The verified ProofWriter answer remains available as the ground-truth label.

---

# 21. Teacher Hidden-State Caching

Because running Teacher and Student simultaneously can cause GPU OOM, use the same general strategy that was developed successfully during the earlier project: run the Teacher offline and cache the required hidden states.

The previous project specifically encountered OOM with simultaneous 3B Teacher/Student inference and solved this by extracting and saving Teacher hidden states before Student training.

For this project:

```text
Teacher inference
       |
       v
Teacher hidden states
       |
       v
Disk cache
       |
       v
Student training
```

The exact hidden layers retained should follow the selected architecture configuration.

Do not cache unnecessary full-model activations if the training objective only needs the final calibrated layers.

---

# 22. Sequence Alignment

Teacher and Student inputs differ:

Teacher:

```text
Demo 1
Demo 2
Target
```

Student:

```text
Target
```

Therefore their token sequences are not directly aligned.

The previous project solved this with dynamic Longest Common Subsequence (LCS) alignment, isolating the shared target/query tokens before hidden-state distillation.

The new implementation should preserve this principle.

Do NOT assume:

```text
teacher_hidden[:, :student_length]
```

corresponds to the Student.

Instead:

1. Tokenize Teacher input.
2. Tokenize Student input.
3. Identify the common target sequence.
4. Map corresponding token positions.
5. Extract Teacher hidden states at those positions.
6. Compare them against Student hidden states at the corresponding positions.

The alignment must be based on actual token identity/position mapping, not hardcoded offsets.

---

# 23. Attention Calibration Architecture

The historical architecture uses a low-rank attention correction:

Standard attention:

```text
S = QKᵀ / sqrt(d)
```

Calibrated attention:

```text
S' = S + gΔ
```

where:

```text
Δ = ABᵀ
```

and:

```text
A = QUq
B = KUk
```

with a query-dependent gate:

```text
g = sigmoid(MLP(Q))
```

The parameters `Uq`, `Uk`, and the gate learn how to modify attention behavior.

The historical project describes this architecture as its adaptation of the Hyper-ICL idea to text reasoning.

---

# 24. Parameter-Efficient Training

The historical implementation used:

- Rank-32 LoRA.
- Final four transformer layers.
- Hidden-state distillation.
- Cross-entropy on final predictions.

This configuration is the starting architecture for the new experiment unless deliberately changed and documented.

The previous report specifies the final-four-layer Rank-32 LoRA configuration.

Do not silently change:

- rank,
- target layers,
- loss,
- tokenizer,
- prompt format,
- dataset condition,
- answer mapping,

between runs.

If a change is made, record it as a separate experimental configuration.

---

# 25. Training Loss

The historical architecture used a dual objective:

```text
L = λ_CE L_CE + λ_MSE L_MSE
```

where:

### Cross-Entropy Loss

Encourages the calibrated Student to produce the correct final answer/output.

### Hidden-State MSE

Encourages the Student hidden representations to approximate the Teacher's corresponding hidden representations.

The exact weighting must be explicitly stored in the experiment configuration.

Never silently change loss weights between runs.

---

# 26. Critical Target-Mapping Rule

The previous project experienced a severe silent target-mapping bug caused by hardcoded answer mappings such as:

```python
["A", "B", "C", "D", "E", "F"]
```

This was incompatible with datasets whose labels were `True`, `False`, `Yes`, `No`, etc., and caused training to reward incorrect behavior. The previous report documents this failure.

The new ProofWriter CWA experiment must therefore use an explicit binary mapping:

```text
True  -> correct True target
False -> correct False target
```

No generic multiple-choice mapping.

No fallback index.

No silent conversion.

Every target label must be validated before training.

---

# 27. Phase 2 Training Rules

The Student must remain zero-shot.

This means:

### Correct

```text
Student:
Target context
Target question
```

### Incorrect

```text
Student:
Demonstration 1
Demonstration 2
Target
```

The Student must never receive the demonstrations during training.

Otherwise the resulting model would not be a demonstration-free calibrated model.

---

# 28. Phase 3 — Evaluation

Phase 3 answers the main research question.

Compare:

### Baseline

Original, untrained:

```text
Qwen2.5-3B
zero-shot
```

against:

### Experimental model

```text
Qwen2.5-3B
trained attention-calibration adapter
zero-shot
```

Both receive exactly the same zero-shot target prompt.

No demonstrations are used for either model.

---

# 29. Phase 3 Evaluation Set

The primary evaluation set is the held-out Golden Test set.

The test set must contain records that satisfied:

```text
Zero-shot original = WRONG
Few-shot Teacher = CORRECT
```

before training.

This makes the evaluation directly test whether the calibrated Student can recover behavior that the untrained zero-shot model could not produce.

---

# 30. Phase 3 Comparison

For each held-out Golden Record, record:

```text
id
gold_answer

original_zero_shot_answer
original_zero_shot_correct

calibrated_zero_shot_answer
calibrated_zero_shot_correct
```

Then classify:

### Recovery

```text
Original: WRONG
Calibrated: CORRECT
```

This is the primary positive result.

### Still Failed

```text
Original: WRONG
Calibrated: WRONG
```

No recovery.

### Regression

```text
Original: CORRECT
Calibrated: WRONG
```

For the Golden Test set this should normally be absent by construction, but the same evaluation should also be performed on a broader held-out set to measure regression.

### Preserved Correctness

```text
Original: CORRECT
Calibrated: CORRECT
```

---

# 31. Recommended Phase 3 Metrics

At minimum report:

## Baseline zero-shot accuracy

```text
correct_original / total
```

## Calibrated zero-shot accuracy

```text
correct_calibrated / total
```

## Golden Record recovery rate

```text
original_wrong -> calibrated_correct
------------------------------------
total Golden Test records
```

## Regression rate

```text
original_correct -> calibrated_wrong
------------------------------------
total evaluation records
```

## Absolute improvement

```text
calibrated_accuracy - original_accuracy
```

All metrics must be calculated from persisted per-example records.

---

# 32. Broader Evaluation

The Golden Test set is the primary test because it directly tests the hypothesis.

However, if computationally feasible, a secondary evaluation should be run on a held-out ProofWriter set that was not selected based on the Golden condition.

This distinguishes:

```text
Can the model recover Golden Records?
```

from:

```text
Did the calibration actually improve general zero-shot reasoning?
```

The broader evaluation is secondary, not a prerequisite for completing the three-phase experiment.

---

# 33. What Counts as Success

A successful result is NOT defined as achieving a particular benchmark percentage.

The primary evidence of success is:

```text
Original zero-shot:
fails

Few-shot Teacher:
succeeds

Calibrated zero-shot:
succeeds
```

on held-out examples.

The strongest result would be a statistically and practically meaningful increase in zero-shot performance while maintaining a low regression rate.

---

# 34. What Does NOT Count as Success

The following are not sufficient:

### Training-set memorization

If the calibrated model succeeds only on examples it trained on, this is not evidence of generalization.

### Evaluating on contaminated records

Records whose original zero-shot failure resulted from incorrect prompting must not be used as valid Golden Records.

### Giving demonstrations to the Student

That changes the experimental condition.

### Giving the target answer to the Teacher

That invalidates the Teacher condition.

### Using generated Teacher reasoning without validation

A bad Teacher target can poison the distillation process.

### Changing prompts between baseline and calibrated evaluation

The baseline and calibrated models must receive the same zero-shot prompt.

---

# 35. No Qwen-Based Demonstration Filtering

Do NOT do this:

```text
Generate demonstration
        |
Ask Qwen whether it can solve demonstration
        |
Keep/discard demonstration
```

This is unnecessary and introduces another model-dependent selection process.

ProofWriter already supplies verified demonstrations.

Use the dataset's verified proof/reasoning.

---

# 36. No BBH Reproduction

This project does not repeat the old BBH experiment.

The old BBH setup introduced excessive evaluation/parsing complexity and produced unreliable results.

The new dataset should minimize answer parsing:

```text
True / False
```

The project is about testing the technique, not reproducing the old benchmark.

---

# 37. No Shot-Count Sweep

The initial experiment is fixed at:

```text
2-shot Teacher
0-shot Student
```

Do not introduce:

```text
1-shot
3-shot
5-shot
8-shot
```

unless the core three-phase experiment has already been completed and a later experiment explicitly investigates shot count.

The purpose of this project is to establish a clean experimental result, not to expand the experiment indefinitely.

---

# 38. No Demonstration-Sweep Experiment

Do not simultaneously introduce:

- Different demonstration-selection strategies.
- Different shot counts.
- Different model sizes.
- Different datasets.
- Different attention architectures.

The initial experiment should change as few variables as possible.

---

# 39. Experiment Reproducibility

Every run must record a configuration containing at least:

```yaml
model:
  name:
  revision:

dataset:
  name:
  configuration:
  split:
  depth:

prompt:
  template_version:
  num_demonstrations: 2

generation:
  temperature:
  top_p:
  max_new_tokens:
  seed:

training:
  adapter:
  rank:
  target_layers:
  learning_rate:
  epochs:
  batch_size:
  gradient_accumulation:
  ce_weight:
  mse_weight:

evaluation:
  parser_version:
```

The exact configuration used to produce every result must be recoverable.

---

# 40. File/Data Organization

A clean implementation should separate the three phases.

Suggested structure:

```text
project/
│
├── README.md
├── PROJECT_SPEC.md
│
├── configs/
│   └── experiment.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── demonstrations/
│   ├── phase1/
│   └── phase2/
│
├── phase1/
│   ├── prepare_dataset.py
│   ├── run_zero_shot.py
│   ├── run_few_shot.py
│   ├── parse_outputs.py
│   ├── classify_records.py
│   └── build_golden_records.py
│
├── phase2/
│   ├── cache_teacher.py
│   ├── align_sequences.py
│   ├── attention_calibration.py
│   ├── train.py
│   └── checkpoints/
│
├── phase3/
│   ├── evaluate_baseline.py
│   ├── evaluate_calibrated.py
│   └── compare_results.py
│
├── results/
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
│
└── logs/
```

The exact structure can differ, but Phase 1, Phase 2, and Phase 3 must remain logically separated.

---

# 41. Phase 1 Required Artifacts

Phase 1 is complete only when it produces:

```text
all_records.jsonl
golden_records.jsonl
both_failed.jsonl
both_passed.jsonl
regressions.jsonl
phase1_config.json
```

`all_records.jsonl` is authoritative.

The other files can be derived from it.

Every record must retain its original ID.

---

# 42. Phase 2 Required Artifacts

Phase 2 should produce:

```text
golden_train.jsonl
golden_validation.jsonl
golden_test.jsonl

teacher_hidden_states/
student_training_config.json

checkpoint/
training_metrics.json
```

The Golden split files must preserve the original Phase 1 IDs.

---

# 43. Phase 3 Required Artifacts

Phase 3 should produce:

```text
baseline_results.jsonl
calibrated_results.jsonl
comparison.jsonl
summary.json
```

Every comparison row must contain the original question ID.

This allows a human to trace:

```text
Final result
    ↓
Question ID
    ↓
Phase 1 record
    ↓
Original Teacher/Student outputs
    ↓
Gold proof
```

without rerunning inference.

---

# 44. Agent Safety Rules

Coding agents working on this project must follow these rules.

## Rule 1 — Do not alter experimental conditions silently.

Any change to prompts, labels, demonstrations, model configuration, or evaluation logic must be explicit.

## Rule 2 — Never put target gold information into the Teacher or Student prompt.

The gold target answer/proof is evaluation data only.

## Rule 3 — Never give demonstrations to the Student.

The Student is always zero-shot.

## Rule 4 — Never classify a record using only generated text without checking the gold dataset label.

## Rule 5 — Never hardcode a generic answer-index mapping.

ProofWriter CWA is binary:

```text
True
False
```

## Rule 6 — Never overwrite raw Phase 1 outputs.

Raw generation results are experimental evidence.

## Rule 7 — Never discard failed examples.

A failed example can be:

```text
Golden
Both Fail
Regression
```

and all categories are informative.

## Rule 8 — Preserve IDs throughout every transformation.

## Rule 9 — Do not rerun Qwen simply to reconstruct information that was already generated.

## Rule 10 — Do not modify the test set after training begins.

---

# 45. Exact Conceptual Pipeline

The entire project can be summarized as:

```text
                    PROOFWRITER
                         |
                         v
                Select target questions
                         |
             +-----------+-----------+
             |                       |
             v                       v
       ZERO-SHOT STUDENT       2-SHOT TEACHER
             |                       |
       Target only             Demo 1
                               Demo 2
                               Target
             |                       |
             v                       v
        ZS reasoning             FS reasoning
        ZS answer                FS answer
             |                       |
             +-----------+-----------+
                         |
                         v
                  Compare to gold
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       GOLDEN         BOTH FAIL     BOTH PASS
       ZS wrong       both wrong    both correct
       FS correct
          |
          v
     Golden Train/Test
          |
          v
   ATTENTION CALIBRATION
   + HIDDEN DISTILLATION
          |
          v
   CALIBRATED ZERO-SHOT
          |
          v
       PHASE 3
          |
          v
 Compare against original
      zero-shot model
```

---

# 46. The Three Phases in One Sentence Each

### Phase 1 — Discovery

> Find and persist examples where Qwen2.5-3B fails zero-shot but succeeds with two verified reasoning demonstrations.

### Phase 2 — Training

> Train the zero-shot Student using attention calibration and hidden-state distillation from the two-shot Teacher, using only Phase 1 Golden Records.

### Phase 3 — Evaluation

> Test whether the calibrated model can solve held-out Golden Records zero-shot that the original model could not solve zero-shot.

---

# 47. Final Experimental Contract

The project must satisfy the following invariant:

### Teacher

```text
2 solved demonstrations
+
target context
+
target question
=
Teacher generates reasoning + answer
```

### Student

```text
target context
+
target question
=
Student generates reasoning + answer
```

### Training

```text
Teacher hidden behavior
        ↓
attention calibration + distillation
        ↓
zero-shot Student
```

### Evaluation

```text
Original zero-shot
        VS
Calibrated zero-shot
```

### Golden Record

```text
Original zero-shot = WRONG
Teacher two-shot   = CORRECT
```

### Target gold information

```text
AVAILABLE TO:
- evaluator
- dataset
- loss computation where appropriate

NOT AVAILABLE IN:
- Teacher target prompt
- Student prompt
- Teacher target input
- Student target input
```

This is the core experimental boundary. Violating it invalidates the corresponding experiment.

---

# 48. Historical Context

The previous project established several useful engineering lessons:

- Qwen2.5-3B-Instruct was selected because it provided a practical balance between reasoning ability and local fine-tuning cost.
- Running Teacher and Student simultaneously caused OOM.
- Offline Teacher hidden-state caching solved that memory problem.
- Teacher/Student sequence mismatch required dynamic alignment.
- Incorrect answer mapping can silently destroy training.
- Prompt-template inconsistency can manufacture false Golden Records.
- The old BBH experiment was too difficult to evaluate reliably.

These lessons should be preserved, but previous numerical results must not be treated as evidence for the new experiment.

---

# 49. Project Philosophy

This is not intended to establish a large formal research laboratory or to exhaustively benchmark every possible configuration.

It is a practical technique-testing project.

The desired output is a reliable experimental record showing:

1. What technique was tested.
2. Under what conditions it was tested.
3. What examples the technique helped.
4. What examples it did not help.
5. Whether improvements survive held-out evaluation.
6. What engineering problems were encountered.
7. What other engineers working with small models can learn from the result.

The most important property is therefore **experimental cleanliness and traceability**, not an impressive headline accuracy.

# Critical Project Requirements

## 51. Phase 1 Dataset

Phase 1 must use exactly **500 ProofWriter CWA instances**:

* 250 with gold answer `True`
* 250 with gold answer `False`

The 500-instance pool must be selected and saved **before inference**.

The 250/250 balance applies to the **initial Phase-1 target pool**, not to the Golden Records.

After Teacher/Student evaluation, preserve the natural Golden Record distribution. Do not artificially balance or discard Golden Records.

Every instance must retain its original dataset ID and gold proof.

---

## 52. Architecture

The new model modification must be **strictly based on the HyperICL attention-calibration methodology**.

The core mechanism is:

```text
S' = S + gΔ

Δ = ABᵀ

A = QUq
B = KUk

g = sigmoid(MLP(Q))
```

The purpose is to learn a compact attention-calibration mechanism that allows the zero-shot Student to reproduce useful behavior demonstrated by the few-shot Teacher.

The old project's code may be reused as an **engineering reference**, especially for:

* Qwen loading
* hidden-state extraction
* LCS/token alignment
* memory handling
* LoRA/parameter-efficient training
* training/evaluation infrastructure

However, the old implementation is **not** the architectural source of truth.

**HyperICL is the architectural source of truth.**

Do not replace the HyperICL-based mechanism with ordinary LoRA fine-tuning.

---

## 53. Hidden-State Persistence

Teacher hidden states required for distillation must be:

1. Generated offline.
2. Saved to disk.
3. Persisted in Google Drive.
4. Reusable by Phase 2 without rerunning Teacher inference.

Do not keep Teacher hidden states only in RAM.

Each saved hidden-state artifact must retain enough metadata to associate it with:

* original question ID
* relevant token positions/alignment
* layer(s)
* tensor dimensions
* model/configuration used

If a valid hidden-state file already exists, the pipeline should reuse it instead of recomputing it.

---

## 54. Compute / Development Workflow

### Local machine

The coding agent runs in local VS Code.

It writes and maintains the project code and pushes it to GitHub.

### GitHub

GitHub is the source of truth for code.

### Google Colab

Colab is the compute environment.

The user will:

1. Pull the GitHub repository into Colab.
2. Mount Google Drive.
3. Run the phase script.
4. Save all persistent outputs to Google Drive.

### Google Drive

Google Drive stores persistent experiment artifacts:

* Phase-1 outputs
* Golden Records
* Teacher outputs
* Student outputs
* Teacher hidden states
* Training checkpoints
* Phase-3 results
* Logs/configuration

Colab's local filesystem must be treated as temporary.

---

## 55. One Main Python File Per Phase

The project must have exactly one primary executable Python file for each phase (:if more than one is absolutely necessary consultme)

```text
phase1.py
phase2.py
phase3.py
```

### phase1.py

Runs:

```text
500 balanced targets
        ↓
2-shot Teacher
0-shot Student
        ↓
compare with gold
        ↓
classify every record
        ↓
save all results + Golden Records
```

### phase2.py

Runs:

```text
Golden Records
        ↓
saved Teacher hidden states
        ↓
HyperICL-based attention calibration
        ↓
distillation/training
        ↓
save calibrated model/checkpoint
```

### phase3.py

Runs:

```text
original Qwen zero-shot
        VS
calibrated Qwen zero-shot
        ↓
held-out evaluation
        ↓
save per-question comparison + metrics
```

The scripts must support resuming from already-completed work so that a Colab disconnect does not force unnecessary recomputation.

---

## 56. Experimental Conditions — Non-Negotiable

### Teacher

```text
2 verified solved demonstrations
+
target context
+
target question
```

The demonstrations contain:

```text
context
question
verified reasoning/proof
verified answer
```

The **target does NOT contain its reasoning or answer**.

### Student

```text
target context
+
target question
```

The Student receives:

* no demonstrations
* no target reasoning
* no target answer

Both models generate reasoning and a final answer.

---

## 57. Golden Record

A Golden Record is strictly:

```text
Teacher/few-shot = CORRECT
Student/zero-shot = WRONG
```

All Phase-1 records must be saved, including:

* Golden Records
* both wrong
* both correct
* Student correct / Teacher wrong

Do not rerun Qwen merely to reconstruct these categories.

---

## 58. Phase Separation

### Phase 1

Discover and save Golden Records.

### Phase 2

Train only on Golden Records using the HyperICL-based architecture.

### Phase 3

Compare the original zero-shot model against the trained/calibrated zero-shot model on held-out data.

The Student remains **zero-shot in all three phases**.

# End

# End of Project Specification