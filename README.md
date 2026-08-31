# LoRA fine-tuning on a Mac: does it beat prompting?

Reproducible experiment behind the article
[LoRA fine-tuning an LLM on your own data](https://jguillaumesio.com/blog/).

**Task**: given one line of text (log line, support message, social post, in
English or French), decide whether it contains personal data and which kinds,
answering in strict JSON:

```json
{"pii": true, "types": ["email", "name"]}
```

Types: `email`, `phone`, `name`, `iban`, `address`, `dob`.

**Question**: does a LoRA fine-tune actually beat the same base model prompted
zero-shot and few-shot, and by how much?

Everything runs locally on Apple Silicon with [MLX](https://github.com/ml-explore/mlx-lm).
No cloud, no API keys, no cost. Hardware used: MacBook, Apple M5, 16 GB.

## Results

Base model: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, temperature 0 in
every run, identical prompts across modes.

### Held-out test set (400 rows)

| Metric | Base, zero-shot | Base, few-shot (6) | LoRA fine-tuned |
|---|---|---|---|
| Accuracy | 66% | 66% | **95%** |
| Precision | 0.639 | 0.610 | **0.926** |
| Recall | 0.686 | 0.840 | **0.974** |
| F1 | 0.662 | 0.707 | **0.950** |
| False positives | 75 | 104 | **15** |
| Missed PII lines | 61 | 31 | **5** |
| Valid JSON | 100% | 100% | 100% |
| Seconds per line | 0.83 | 1.67 | 0.91 |

### Per-type F1

| PII type | Zero-shot | Few-shot | LoRA |
|---|---|---|---|
| email | 0.745 | 0.782 | **1.000** |
| phone | 0.628 | 0.575 | **0.983** |
| name | 0.531 | 0.663 | **0.855** |
| iban | 0.358 | 0.194 | **0.950** |
| address | 0.500 | 0.597 | **0.914** |
| dob | 0.383 | 0.366 | **0.875** |

### Out-of-distribution set (30 hand-written lines)

Written by hand, in phrasings the training data never contained, to check the
model is not just fitting the corpus it was trained on.

| Metric | Zero-shot | Few-shot | LoRA |
|---|---|---|---|
| Accuracy | 80% | 90% | **100%** |
| False positives | 2 | 1 | **0** |
| Missed PII lines | 4 | 2 | **0** |

### Training cost

| | |
|---|---|
| Trainable parameters | 0.34% of the model |
| Peak memory | 6.0 GB |
| Wall clock | ~45 minutes (500 iterations) |
| Adapter size | 42 MB |
| Money | 0 EUR |

Validation loss: 3.537 at iteration 1, 0.508 at 250, 0.491 at 500. Training was
stopped at 500 because the curve had flattened: the previous 250 iterations
bought a 3% improvement.

## Follow-up: one artifact, two sets

A reader pointed out that the drop comparison in the original write-up was not
what it looked like, and he was right. The "5 point drop" for the fine-tune
compared **two different adapters** (v1-trained on the v1 set against v2-trained
on the v2 set), while the prompted rows were genuinely the same method twice.

Holding the artifact fixed instead:

| v1-trained adapter | Accuracy | F1 | False positives | Misses |
|---|---|---|---|---|
| on the v1 synthetic set | 100% | 1.000 | 0 | 0 |
| on the v2 real set | **67%** | 0.701 | 93 | 39 |

```bash
.venv/bin/python evaluate.py --mode lora \
  --adapter-path ./adapters_v1_synthetic --limit 400 --tag _v1adapter_realset
```

Every row below is now one method or artifact measured twice, and the drops rank
by how much each had been fitted to the v1 distribution:

| Method, held fixed | v1 set | v2 set | Drop |
|---|---|---|---|
| Adapter trained on v1 data | 100% | 67% | **33** |
| Few-shot, 6 v1-shaped examples | 94% | 66% | **28** |
| Zero-shot, no examples | 88% | 66% | **22** |

### Does the few-shot example pool contaminate the test set?

The six hand-written `FEW_SHOT` examples in `evaluate.py` are instances of the v1
generator's own templates, so the prompted arm carried v1-shaped hints. `FEW_SHOT_V2`
swaps them for six drawn from `data/train.jsonl` (same count, same 3/3 balance, same
three types, each verified absent from the test set):

```bash
.venv/bin/python evaluate.py --mode few-shot --few-shot-set v2 --limit 400 --tag _v2examples
```

| On the v2 real set (n=400) | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Few-shot, v1 hand-written examples | 66.2% | 0.707 | 0.610 | 0.840 |
| Few-shot, v2 in-domain examples | 67.7% | 0.705 | 0.634 | 0.794 |

1.5 points, which is nothing: both arms ran on the same rows, so use a paired test.

```bash
python3 mcnemar.py
```

| Comparison | Discordant pairs | Exact p |
|---|---|---|
| few-shot v1 vs v2 examples | 21 / 27 | 0.4709 (not significant) |
| zero-shot vs few-shot v1 | 53 / 54 | 1.0000 (not significant) |
| few-shot v2 vs LoRA | 4 / 113 | < 0.0001 (significant) |

So the examples were worth 6 points on the set that shared their provenance and
nothing on the set that did not, and matching the provenance recovers nothing
either: six examples of any origin cannot express this task. The fine-tune's
advantage, meanwhile, is a 113-to-4 split.

Full write-up: [A reader read my benchmark better than I did](https://jguillaumesio.com/blog/llm-benchmark-leakage-few-shot/)

## The finding worth stealing

An earlier version of this experiment used 800 examples I generated from my own
templates. On that data the fine-tune scored 100% and few-shot prompting scored
94%, and the honest conclusion was "fine-tuning barely earns its keep".

On the real data in this repository, the same fine-tune beats the same
prompting baselines by 29 accuracy points, and few-shot prompting is *worse*
than zero-shot on precision (104 false positives against 75) because six
examples cannot teach the boundary between an IBAN and an invoice number across
two languages and two writing styles.

The benchmark chose the conclusion. A weak dataset made fine-tuning look
pointless.

## Dataset construction

The dataset is not shipped here: `build_dataset.py` rebuilds it from two
permissively licensed public sources (see NOTICE). Only the 30 hand-written
out-of-distribution lines, which are original to this repository, are included.

Positives come from `DataikuNLP/kiji-pii-training-data` (Apache-2.0), negatives
from `witfoo/syslog-to-artifact` (Apache-2.0). Two other candidates were
rejected on purpose:

- **ai4privacy/pii-masking-200k**: dual license, free only for individuals and
  companies of three staff or fewer, so not safe for commercial use.
- **LogHub**: licensed for research or academic work only, and several of its
  systems (BGL, Mac, Linux, OpenSSH) contain real usernames, home directory
  paths and email addresses. Labeling those lines "contains no PII" would have
  taught the model the exact opposite of the task.

### Avoiding the style shortcut

If every positive were prose and every negative a log line, the model would
separate the classes by writing style and score well without learning anything
about personal data. So all four quadrants exist:

| | Positive (has PII) | Negative (no PII) |
|---|---|---|
| **Prose** | kiji sentences | same sentences, PII replaced by role words |
| **Log** | real syslog with real PII injected | real syslog, untouched |

A second leak appeared inside that fix: filler phrases like "the customer" only
ever appeared in negatives, so they became a giveaway. 10% of the rows are
therefore *hybrid* positives, using the same filler vocabulary while keeping one
real PII value.

### Data defects found in the public sources

- Dropping every row that mentions out-of-scope PII (SSN, passport, credit card)
  costs 75% of kiji. Those spans are neutralized instead, recovering 4,180 rows
  to 15,810.
- Kiji's `coreferences` field is empty on every row, so an unannotated later
  mention ("Dubois a signé" after "Alice Dubois" was masked) survives into a
  would-be negative. Every negative is screened against a name vocabulary built
  from the corpus itself.
- 4.7% of kiji's French rows are encoding-corrupted: accented characters arrive
  as NUL bytes, so `étude` is stored as `\x00tude`. Those rows are dropped.
- ~4% of the syslog corpus carries account names in prose ("Accepted password
  for johndoe"). Dropped wholesale rather than mislabeled as PII-free.

## Reproducing

```bash
python3 -m venv .venv
.venv/bin/pip install mlx-lm datasets

# 1. rebuild the dataset (downloads both public corpora)
.venv/bin/python build_dataset.py

# 2. baselines, before any training
.venv/bin/python evaluate.py --mode zero-shot --limit 400
.venv/bin/python evaluate.py --mode few-shot  --limit 400

# 3. train the adapter
.venv/bin/python -m mlx_lm lora \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --train --data ./data \
  --fine-tune-type lora \
  --batch-size 4 --num-layers 16 --iters 500 \
  --learning-rate 1e-5 --max-seq-length 512 \
  --mask-prompt --grad-checkpoint \
  --steps-per-report 50 --steps-per-eval 250 --save-every 500 \
  --val-batches 25 --seed 42 \
  --adapter-path ./adapters 2>&1 | tee training.log

# 4. evaluate the fine-tuned model and build the tables
.venv/bin/python evaluate.py --mode lora --adapter-path ./adapters --limit 400
.venv/bin/python evaluate.py --mode lora --adapter-path ./adapters \
  --test-file ood_test.jsonl --tag _ood
.venv/bin/python report.py
```

### Why these hyperparameters

- `--mask-prompt`: compute the loss on the JSON answer only. Without it the
  model spends its capacity learning to predict the input line, which dwarfs
  the label.
- `--batch-size 4`, `--num-layers 16`: chosen after measuring a 4.8 GB peak at
  batch 1 with 8 layers, which left plenty of headroom on a 16 GB machine.
- `--learning-rate 1e-5`: 1e-4 oscillates on small datasets, 1e-6 barely moves.
- Evaluate with `--adapter-path`, not a fused model: there is an open mlx-lm
  issue where fusing can lose behavior the dynamic adapter still shows.

### A Mistral-specific gotcha

Mistral's chat template rejects a standalone `system` role and requires strictly
alternating user/assistant turns. The instructions are prepended to the user
turn instead, in both the dataset builder and the eval harness, so training and
inference see byte-identical formatting. Fixing this in only one of the two
places produces an adapter that looks broken for no visible reason.

## Files

| File | What it does |
|---|---|
| `build_dataset.py` | Rebuilds train/valid/test from the two public corpora |
| `evaluate.py` | Runs one mode over a test set, writes metrics and predictions |
| `report.py` | Assembles the comparison tables into `results/summary.md` |
| `data/ood_test.jsonl` | 30 hand-written out-of-distribution lines |
| `results/` | Metrics for all three modes, plus per-line predictions on the OOD set |
| `training.log` | The actual training run behind the numbers above |

## License

Code in this repository: MIT, see `LICENSE`.
Third-party data attribution: see `NOTICE`.
