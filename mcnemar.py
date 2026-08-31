#!/usr/bin/env python3
"""Paired significance tests on two prediction logs.

Arms evaluated on the same test set are paired, so comparing their accuracies
against an independent-proportion margin of error overstates what you can claim.
McNemar's test looks only at the rows where the two arms disagree, which is the
part that carries information.

Usage:
    python mcnemar.py results/few-shot_predictions.jsonl \
                      results/few-shot_v2examples_predictions.jsonl
    python mcnemar.py            # runs the three comparisons from the article
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent


def correctness(path):
    """True/False per row: did the arm get the pii decision right?"""
    out = []
    for line in open(HERE / path, encoding="utf-8"):
        r = json.loads(line)
        out.append(r["gold"]["pii"] == r["pred"]["pii"])
    return out


def mcnemar(path_a, path_b):
    a, b = correctness(path_a), correctness(path_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if not x and y)
    m = only_a + only_b
    if m == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        # exact two-sided binomial on the discordant pairs
        p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return {
        "n": n,
        "acc_a": round(sum(a) / n, 4),
        "acc_b": round(sum(b) / n, 4),
        "only_a_right": only_a,
        "only_b_right": only_b,
        "p": p,
    }


DEFAULT = [
    ("few-shot, v1 examples", "results/few-shot_predictions.jsonl",
     "few-shot, v2 examples", "results/few-shot_v2examples_predictions.jsonl"),
    ("zero-shot", "results/zero-shot_predictions.jsonl",
     "few-shot, v1 examples", "results/few-shot_predictions.jsonl"),
    ("few-shot, v2 examples", "results/few-shot_v2examples_predictions.jsonl",
     "LoRA v2 adapter", "results/lora_predictions.jsonl"),
]


def main():
    pairs = DEFAULT
    if len(sys.argv) == 3:
        pairs = [(sys.argv[1], sys.argv[1], sys.argv[2], sys.argv[2])]
    for label_a, file_a, label_b, file_b in pairs:
        r = mcnemar(file_a, file_b)
        verdict = "significant" if r["p"] < 0.05 else "not significant"
        print(f"{label_a} ({r['acc_a']:.3f})  vs  {label_b} ({r['acc_b']:.3f})   n={r['n']}")
        print(f"   discordant pairs: {r['only_a_right']} / {r['only_b_right']}"
              f"   exact p = {r['p']:.4f}   {verdict}\n")


if __name__ == "__main__":
    main()
