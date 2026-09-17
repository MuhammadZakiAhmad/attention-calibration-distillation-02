# Project: Attention Calibration Distillation 02

## Overview
- **Goal**: Determine whether attention calibration + hidden-state distillation can train Qwen2.5-3B-Instruct to solve ProofWriter CWA instances zero-shot that it fails zero-shot, but solves when given two verified solved demonstrations. Measure recovery on held-out Golden Records with low regression and full traceability.
- **Stack**: Python, PyTorch, HuggingFace Transformers + Datasets + PEFT, Google Colab (compute), Google Drive (artifact persistence), local VS Code (authoring), GitHub (source of truth).
- **Started**: 2026-09-17 11:48
- **Project Status**: ACTIVE

## Constraints
- Spec file: `Attention Calibration Distillation — Three-Phase Experimental Project Specification.md` (58 sections) is the contract. `HyperICL.pdf` is the architectural source of truth.
- §26: binary label mapping only — `True` / `False`. No generic A/B/C/D index mapping, no fallback index, no silent conversion.
- §44 Agent Safety Rules (10): no silent condition changes, no gold target info in prompts, Student always zero-shot, classify only against gold label, no generic answer-index mapping, never overwrite raw Phase 1 outputs, never discard failed examples, preserve IDs through every transform, never rerun Qwen to reconstruct already-generated data, never modify test set after training begins.
- §51: exactly 500 ProofWriter CWA instances in the Phase-1 pool, 250 `True` / 250 `False`, selected and saved before inference. Golden Record distribution stays natural — do not balance after evaluation.
- §55: exactly one primary executable Python file per phase — `phase1.py`, `phase2.py`, `phase3.py`. Consolidate rather than split without consulting the user.
- §37/§38: no shot-count sweep (fixed 2-shot Teacher / 0-shot Student), no demonstration sweep, no model-size sweep, no dataset sweep.
- §18: split Golden Records by theory ID, not question ID.
- §54: Colab local filesystem is temporary. All persistent artifacts go to Google Drive.
- §10: Teacher and Student must both use the official Qwen chat template. Their only difference is presence/absence of the two demonstrations.

---

## Project Log

### Achievements
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {what} — {how} — {where} -->

[S1-2026-09-17-1148] Phase 1 dataset source identified and schema decoded — inspected `arqa39/proofwriter-source` on the HF Hub; one row per theory with `id`, `theory` (context), `triples`, `rules`, and a `questions` dict carrying `question`/`answer`/`QDep`/`strategy`/`proofsWithIntermediates` — `phase1.py` design, README.md.

[S1-2026-09-17-1148] Confirmed CWA answers are Python `bool` with no `Unknown`, and are perfectly balanced at theory level — sampled 400 theories per config; depth-2 gave 2227 True / 2227 False — so the spec §26 binary mapping is native to the data and the 250/250 pool is easy to fill.

[S1-2026-09-17-1148] ProofWriter proof grammar decoded and a recursive-descent renderer written — the `representation` field is a small S-expression grammar over `tripleN`/`ruleN`/`intN`/`nafN`; the renderer walks it post-order and emits one natural-language step per inference — `phase1.py` (`_tokenize`, `_parse_expr`, `ProofRenderer`).

[S1-2026-09-17-1148] Renderer validated across the whole depth-2 train split — 25,616 questions rendered with **zero** parse failures; the 10,836 failures are all genuine "no proof in dataset" cases, not renderer bugs — local run over `arqa39/proofwriter-source` CWA-depth-2 train (6404 theories).

[S1-2026-09-17-1148] 500-target pool built and persisted before any inference — 250 True / 250 False, QDep>=1, deterministic via `generation.seed`, spanning 494 unique theories, all with a validated gold proof — `data/phase1/pool.jsonl`, `results/phase1/phase1_config.json`.

[S1-2026-09-17-1148] Two fixed demonstrations built from verified dataset proofs — `AttNeg-CWA-D2-1001::Q3` (True) and `AttNeg-CWA-D2-1002::Q4` (False, inv-proof); both theories excluded from the target pool — `data/demonstrations/demonstrations.json`.

[S1-2026-09-17-1148] Leakage and prompt-symmetry properties verified across all 500 targets — assertions that the Student prompt is a verbatim suffix of the Teacher prompt, that the target gold proof never appears in the Teacher prompt, and that demo context never appears in the Student prompt; all four checks returned zero violations — local verification script.

[S1-2026-09-17-1148] Chat-template consistency verified — the tokenization path used by `generate()` (render template to string, then `add_special_tokens=False`) produces token IDs identical to `apply_chat_template(tokenize=True)`; 163 tokens both ways — local test against the real Qwen2.5-3B-Instruct tokenizer. This is the exact failure mode spec §10 warns about.

[S1-2026-09-17-1148] Parser and classifier unit-tested — 9 `parse_answer` cases including `INVALID` handling, last-match-wins, and explicit rejection of `Unknown`/`A` labels; all 4 `classify` outcomes — 0 failures — local test.

[S1-2026-09-17-1148] `classify` stage exercised end-to-end against synthetic generations — produced all six spec §41 artifacts with correct category counts, and confirmed idempotent across repeat runs — throwaway `ACD_ROOT`, since deleted.

[S1-2026-09-17-1148] Resume path verified — with all 500 records already generated, `--stage generate` reports `0 to do` for both conditions and skips model loading entirely — throwaway `ACD_ROOT`, since deleted.

[S1-2026-09-17-1148] Pool integrity re-verified after a user question about search counts — a case-insensitive search of `data/phase1/pool.jsonl` returns 250 hits for "true" but 500 for "false", which is occurrence-based counting, not line-based, and is correct: 250 `"gold_answer": "True"` + 250 `"gold_answer": "False"` + 250 lowercase "false" inside the renderer's False-branch prose (`"...so this claim is false."`). Lines containing each term are 250/250 and the set of distinct `gold_answer` values is exactly `{'True', 'False'}` — local `re.findall` audit over the file.

[S1-2026-09-17-1148] Confirmed no Python booleans leak into persisted records — lowercase `true` occurs 0 times in `pool.jsonl`, so `answer` survives only as the explicit string `"True"`/`"False"` produced by `build_candidates`, never as a serialized native bool (spec §26).

### All Errors Encountered
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] `{error}` — where: {location} — status: {resolved/partial/open} — tried: {what} -->

[S1-2026-09-17-1148] `ProofError: unrecognised proof symbol 'naf1'` — where: `phase1.py` `_atom_key` — status: resolved — tried: inspected `AttNeg-*` theories and found `nafN` (negation-as-failure premises such as "Charlie is not white") stored in the same `intermediates` dict as `intN`, not in a separate table; fixed by routing the `naf` prefix to the intermediate bucket.

[S1-2026-09-17-1148] Rule text silently dropped from rendered proofs, e.g. `From "Harry is quiet.", we can conclude Harry is red..` rendered with no rule and a doubled period — where: `phase1.py` `_parse_expr` / `_conclusion` — status: resolved — tried: dumped the parse tree and found `_parse_expr` wraps every parenthesised form in a group, so a conclusion written `(rule6 % int2)` arrived as `('group',[pct],None)`; `_conclusion` saw a group, not a pct, and returned `rule_text=None`. Fixed by unwrapping single-item arrow-less groups in `_parse_expr` and defensively via a new `_unwrap` helper. Note this failed *silently* — the rendered text still validated against the question, so only inspecting output caught it.

[S1-2026-09-17-1148] `KeyError: 'generation_parameters'` — where: `phase1.py` `stage_classify` — status: resolved — tried: found the line used `zero_shot["generation_parameters"]` where `zero_shot` is the id-keyed dict, not the per-record `zs`; fixed to `zs["generation_parameters"]`. Caught only by the synthetic end-to-end classify test — it would otherwise have surfaced in Colab *after* the expensive GPU generation stage.

[S1-2026-09-17-1148] `No pool found at C:\tmp\acd_test\...` — where: local smoke test — status: resolved — tried: used a Git Bash `/tmp` path as `ACD_ROOT`; Python's `Path.resolve()` maps `/tmp` to `C:\tmp` on Windows while Git Bash maps it elsewhere. Fixed by using a Windows-native path for the throwaway root. Not a code bug, but worth remembering for any local test that sets `ACD_ROOT`.

### Dead Ends (do not retry)
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {what tried} — why: {reason} — retry if: {condition or "never"} -->

[S1-2026-09-17-1148] `tasksource/proofwriter` as the dataset source — why: it is a flattened per-question build (845k rows) with the theory context duplicated per row and theory grouping only implicit in the `id` string; `arqa39/proofwriter-source` groups by theory natively and ships the `triples`/`rules`/`proofsWithIntermediates` tables the renderer needs — retry if: never, unless the source mirror disappears from the Hub.

[S1-2026-09-17-1148] Using `proofDetails` (theory-level) as the proof source — why: it aggregates every fact derivable in the theory across all questions, so it cannot identify which chain belongs to the specific question being asked; the per-question `proofsWithIntermediates` is the correct granularity — retry if: never.

### Decisions Made
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {decision} — why: {reason} — rejected: {alternatives} — reversible: {yes/no} -->

[S1-2026-09-17-1148] Phase 2 architecture = HyperICL attention calibration **plus** rank-32 LoRA on the final 4 transformer layers — why: spec §24 names the historical rank-32/final-4-layer LoRA config as the starting architecture while §52 forbids replacing the HyperICL mechanism with ordinary LoRA; combining satisfies both, and §24 explicitly permits changes when recorded as a separate experimental configuration — rejected: pure-HyperICL-only (cleanest mechanism test, kept as a possible later config), calibration-only-then-LoRA-later — reversible: yes (ablation config).

[S1-2026-09-17-1148] ProofWriter CWA sourced from HuggingFace Hub rather than the official tarball — why: fastest in the Colab workflow, supplies the gold proof field needed for verified demonstrations — rejected: official ProofWriter release tarball + custom parser — reversible: yes. **Exact dataset ID and field names still unverified.**

[S1-2026-09-17-1148] `project_history/` lives at the repository root rather than the skill's default `../project_history/` — why: history should be version-controlled and travel with the code to GitHub and Colab — rejected: `.claude/skills/project_history/` — reversible: yes.

[S1-2026-09-17-1148] Session 1 scope = Phase 1 end-to-end — why: user selection; Phase 1 is a prerequisite for every later phase and its artifacts are the only source of Golden Records — rejected: design-docs-only, scaffold-and-push-only — reversible: no (scope already executed).

[S1-2026-09-17-1148] Dataset = `arqa39/proofwriter-source`, configuration `CWA-depth-2`, split `train` — why: it is an unmodified copy of AI2 ProofWriter that keeps one row per theory (so theory-level grouping is native, satisfying the §18 split rule), and depth-2 is the shallowest config with real multi-hop content — measured 400 theories at depth-2: 2227 True / 2227 False, QDep {0:2140, 1:1334, 2:928, 3:40, ...} — rejected: `tasksource/proofwriter` (flattened, 845k rows, implicit theory grouping), depth-1 (too shallow: 58% of questions are QDep=0), depth-3 (only ~900 QDep>=2 questions per 400 theories) — reversible: yes, but changing it invalidates any collected Phase 1 data.

[S1-2026-09-17-1148] Target eligibility requires `QDep >= 1` **and** a present dataset proof — why: QDep=0 questions are direct fact lookups that zero-shot Qwen almost certainly solves, so they would become Both-Pass records rather than Golden Records; and a question with no proof cannot supply a verified demonstration either — rejected: no depth filter (would dilute the pool with trivial instances) — reversible: yes via `dataset.min_qdep`.

[S1-2026-09-17-1148] Proof rendering uses a full recursive-descent parser of the `representation` grammar, not flat text assembly — why: spec §7/§8 require demonstrations to carry *verified* reasoning, and only the parse tree preserves which rule connects which premises at each step; the grammar is small and regular, and the parser achieved a zero failure rate on 25,616 questions — rejected: flat assembly of `intermediates` in QDep order (loses per-step inference structure), top-level-chain-only (loses nested sub-proofs) — reversible: yes.

[S1-2026-09-17-1148] Demonstrations are a fixed pair, one gold True and one gold False, drawn from two theories that are then excluded from the target pool — why: permitted by §7, maximally comparable across targets since every target sees identical demonstrations, and the theory exclusion satisfies §8's "different theories than the target" requirement by construction — rejected: hash-selected per-target (adds a variance source), difficulty-matched (more selection logic, no clear benefit at this scale) — reversible: yes, but changing it after Phase 2 invalidates the comparison.

[S1-2026-09-17-1148] Generation uses greedy decoding, `max_new_tokens=512` — why: spec §16 requires every generation to be cached and §9 forbids rerunning on parse failure, both of which presuppose deterministic decoding; 512 leaves headroom so deep proof chains do not truncate into a spurious `INVALID` that would pollute the Golden set — rejected: 256 tokens (truncation risk), sampled at temp 0.7 (non-deterministic, contradicts the spec's accounting model) — reversible: yes, but any change requires re-running Phase 1 from scratch.

[S1-2026-09-17-1148] `classify` merges into `phase1_config.json` rather than overwriting the prepare-stage metadata — why: spec §39 requires the run configuration to stay recoverable, and the pool-selection record is part of that evidence — rejected: separate `phase1_prepare_config.json` (a second file per phase, against §55's spirit) — reversible: yes.

[S1-2026-09-17-1148] `stage_generate` determines outstanding work before loading the model — why: loading a 3B model to discover there is nothing to do wastes minutes of Colab time per resume — rejected: loading first and filtering after (simpler, slower) — reversible: yes.

[S2-2026-09-17-1228] Model stays `Qwen/Qwen2.5-3B-Instruct`; the GQA mismatch is **not** sufficient reason to swap models — why: worked the algebra and the 16:2 query/KV head ratio needs no redesign. `Δ_q = (Q_q Uq)(K_g Uk)ᵀ = Q_q Uq Ukᵀ K_gᵀ` for any query head `q` in KV-group `g`, so repeating `Uk`'s output across the group is arithmetically identical to any grouped formulation, because GQA already shares `Uk` within a group. Spec §23's `A = QUq` / `B = KUk` therefore applies verbatim. Swapping models would cost far more than it saves: it orphans the §24 "historical configuration" lineage and invalidates `target_layers` — rejected: switching to a multi-head-attention model with matching Q/K head counts (a real convenience, but it discards the spec's named starting architecture to avoid a one-line `repeat_kv`) — reversible: yes, but expensive (invalidates all Phase 1 generation).

[S2-2026-09-17-1228] `model.revision` pinned to `aa8e72537993ba99e69dfaafa59ed015b17504d1` — why: spec §39 reproducibility. Generation is cached per record, so if the Colab run and the final reported run sat on different commits the cache would silently mix model versions and the whole traceability chain would be void. Cheap now, unrecoverable later — rejected: leaving `main` (the commit moves under us; a re-run months later would not reproduce) — reversible: yes, but only *before* the Colab generation run starts.

[S2-2026-09-17-1228] `training.calibration.distill_layers` frozen to `[32, 33, 34, 35]`, equal to `target_layers` — why: the distillation objective is precisely what the calibrated layers produce, so no other layer's hidden states are needed; §21 explicitly says not to cache full-model activations the objective does not use. Caching one four-layer set instead of the whole model also keeps the Drive artifact small — rejected: caching all 36 layers (spec-permissible but wasteful), caching a separate set of layers from the calibrated ones (would require a second, unexplained extraction) — reversible: no, not without re-extracting the cache.

[S2-2026-09-17-1228] Calibration attaches **post-RoPE**, in raw attention scores, with `Uq` consuming **scaled** Q, and `Uq`/`Uk` zero-initialised — why: §23 defines `S' = S + gΔ`, which only means anything if Δ lives in the same space and units as `S`. Qwen applies RoPE to Q/K *before* the score matmul and scales Q, so a pre-RoPE or unscaled Δ is a different quantity than the paper's. Zero-init makes Δ = 0 at step 0, so the calibrated model starts bit-identical to the base model — which Phase 3's baseline comparison depends on — rejected: pre-RoPE injection (can't express the correction as a score modification), non-zero init (breaks the "starts from baseline" property, and any Phase 3 gain would be uninterpretable) — reversible: no, this is the architecture.

[S2-2026-09-17-1228] The model must run with `attn_implementation: eager` — why: the correction is added to the raw attention scores, so those scores have to be materialised. Flash-attention and SDPA never expose them and cannot be patched without reimplementing the kernel. Cost is bounded because it applies to four layers, not 36 — rejected: `sdpa` (same problem), `flash_attention_2` (fastest, and quietly makes the whole calibration mechanism a no-op) — reversible: no.

[S2-2026-09-17-1228] `L_CE` targets the Teacher's **full** generated output (reasoning + final answer), prompt tokens masked — why: §20 names "teacher reasoning" and "teacher final answer" as the Teacher target-side outputs, so the chain is the target — rejected: supervising only the final-answer line (much weaker signal; the calibrated model would learn to answer without learning the intermediate structure the calibration is meant to transfer). **Carried risk, recorded deliberately**: a Golden Record guarantees the Teacher's *answer* is right, not that its *reasoning* is. Distilling a plausible-but-wrong chain is exactly the silent-wrong failure mode that cost this project an earlier iteration. If Phase 3 shows confident nonsense, this is the first suspect — reversible: yes.

[S2-2026-09-17-1228] `L_MSE` aligns on **target-block positions only** — why: §22 requires the mapping be derived from token identity, and the two prompts already share a byte-identical target block by construction, so the alignment is a clean, well-defined suffix match. Note the sequences *also* share the system block at the front, so a naive LCS over the full sequence returns both regions — the alignment window must be restricted to the target region and asserted on, per §22's "isolate the shared target/query tokens" — rejected: aligning over the whole common subsequence (dilutes the signal with positionally-identical boilerplate that carries nothing worth distilling) — reversible: yes.

### Features
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {feature name} — status: {planned/in-progress/completed} — {description} -->

[S1-2026-09-17-1148] phase1.py — completed — 500-target balanced pool selection, 0-shot Student and 2-shot Teacher generation via the Qwen chat template, ProofWriter proof rendering, answer parsing, classification into Golden/Both-Fail/Both-Pass/Regression, all six Phase-1 artifacts, resume support. Pool + renderer + parser + classifier verified locally; GPU generation stage runs in Colab.

[S1-2026-09-17-1148] phase2.py — planned — Golden-Record split by theory ID, offline Teacher hidden-state caching, LCS token alignment, HyperICL attention calibration + rank-32 LoRA training, checkpoint persistence.

[S1-2026-09-17-1148] phase3.py — planned — original zero-shot vs calibrated zero-shot held-out evaluation, per-question comparison records, summary metrics.

[S1-2026-09-17-1148] configs/experiment.yaml — completed — single experiment configuration holding model, dataset, prompt, generation, training, and evaluation settings per spec §39. Values still to be revisited before Phase 2: `training.calibration.distill_layers`, `training.epochs`, `ce_weight`/`mse_weight`.

[S1-2026-09-17-1148] Drive-root artifact abstraction — completed — every script resolves its output root from `$ACD_ROOT` (falling back to the configured default), so local and Colab runs share one code path per spec §54.

[S1-2026-09-17-1148] ProofWriter proof renderer — completed — recursive-descent parser over the `tripleN`/`ruleN`/`intN`/`nafN` grammar with per-step natural-language emission and conclusion validation against the question. Lives inside `phase1.py` to respect the one-file-per-phase rule.

### Important Files
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {file path} — {description/purpose} -->

[S1-2026-09-17-1148] `Attention Calibration Distillation — Three-Phase Experimental Project Specification.md` — the experimental contract; all conditions, restrictions, and required artifacts.

[S1-2026-09-17-1148] `HyperICL.pdf` — paper defining the architectural source of truth (`S' = S + gΔ`, `Δ = ABᵀ`, `A = QUq`, `B = KUk`, `g = sigmoid(MLP(Q))`).

[S1-2026-09-17-1148] `phase1.py` — the single Phase-1 executable: config loading, ProofWriter proof renderer, dataset/pool/demonstration selection, prompt construction, generation, parsing, classification, artifact persistence, resume logic.

[S1-2026-09-17-1148] `configs/experiment.yaml` — every value consumed by the phase scripts; nothing is hardcoded in code. Changing any value is a new experimental configuration.

[S1-2026-09-17-1148] `README.md` — project overview, quick start, Colab instructions (`ACD_ROOT`), artifact layout, and the design notes behind the dataset/demo/rendering choices.

[S1-2026-09-17-1148] `data/phase1/pool.jsonl` — the 500 selected targets, written before any inference.

[S1-2026-09-17-1148] `data/demonstrations/demonstrations.json` — the two fixed verified demonstrations and the excluded theory IDs.

[S1-2026-09-17-1148] `results/phase1/phase1_config.json` — pool selection evidence (candidate counts, skipped breakdown, demo IDs, excluded theories) plus the full config.

[S1-2026-09-17-1148] `project_history/project_details.md` — this file; append-only project log.

---

## Session Index
| ID | Date | Status | Working On |
|----|------|--------|------------|
| S1-2026-09-17-1148 | 2026-09-17 | PAUSED | Phase 1 built + verified locally; Colab GPU run not yet executed |
| S2-2026-09-17-1228 | 2026-09-17 | ACTIVE | Pin model.revision + freeze training.calibration.distill_layers before hidden-state caching |
