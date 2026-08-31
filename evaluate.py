#!/usr/bin/env python3
"""Evaluate PII detection on the held-out test set.

Three modes, all at temperature 0 with the same chat template:
    zero-shot  : base model, system prompt only
    few-shot   : base model, system prompt + 6 in-context examples
    lora       : base model + trained adapter (--adapter-path)

Reports: accuracy, precision/recall/F1 on the "contains PII" decision,
false-positive rate on hard negatives, per-type F1, valid-JSON rate,
and wall-clock. Writes results/<mode>.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).parent
TYPES = ["email", "phone", "name", "iban", "address", "dob"]
SYSTEM_PROMPT = (
    "You are a PII detector. Given one line of text, reply with strict JSON "
    'only: {"pii": true/false, "types": [...]}. Allowed types: "email", '
    '"phone", "name", "iban", "address", "dob". Use an empty list when '
    '"pii" is false. No other text.'
)

FEW_SHOT = [
    ("2026-03-02T10:11:12Z INFO deploy finished commit=a1b2c3d4e5 in 42s",
     {"pii": False, "types": []}),
    ("Please send the invoice to marie.dupont@example.com",
     {"pii": True, "types": ["email"]}),
    ("Refund processed for order ORD-448120, amount 49.90 EUR.",
     {"pii": False, "types": []}),
    ("Le client a appele depuis le 06 12 34 56 78 pour un paiement refuse.",
     {"pii": True, "types": ["phone"]}),
    ("The staging API key sk_ab12CD34ef56GH78ij90KL12 was rotated.",
     {"pii": False, "types": []}),
    ("Ticket opened by Sophie Bernard regarding order ORD-119284.",
     {"pii": True, "types": ["name"]}),
]

# Same size and the same 3 negative / 3 positive balance as FEW_SHOT above, and the
# same three PII types (email, phone, name), but every example is drawn from
# data/train.jsonl instead of being written by hand. Each was checked against
# data/test.jsonl for exact-string absence.
#
# Why this exists: a reader pointed out that the example pool is part of the prompted
# method's input, so kinship between the pool and the test set inflates the prompted
# arm and not the fine-tuned one. FEW_SHOT above is hand-written in the shapes of the
# v1 synthetic generator, which is exactly that kinship. This set swaps the pool for
# one whose provenance matches the v2 test set, holding everything else fixed.
FEW_SHOT_V2 = [
    ("<187>315106: WCFREP01: Jul 26 18:40:55.494 CDT: %LINK-3-UPDOWN: Interface "
     "GigabitEthernet2/0/19, changed state to down",
     {"pii": False, "types": []}),
    ("Jul 26 12:38:30 oakrhelv002 filebeat: 2024-07-26T12:38:30.317-0500 INFO "
     "log/harvester.go:333 File is inactive: /var/log/btmp-20240701.",
     {"pii": False, "types": []}),
    ("Bonjour, je suis le client. Mon numero de TVA est la reference fiscale. Vous "
     "pouvez consulter mes factures sur le wiki interne.",
     {"pii": False, "types": []}),
    ("Jul 26 06:55:42 oakrhelv002 filebeat: retryer: send unwait signal to consumer "
     "user=emily.nguyen@mail.com.au",
     {"pii": True, "types": ["email"]}),
    ("2 733177541390 eni-09380904df4353210 - - - - - - - 1722000541 1722000572 - "
     "NODATA callback=(406) 809-7623",
     {"pii": True, "types": ["phone"]}),
    ("the customer MacIntyre our regional hub, the region the delivery zone the "
     "support address | the hotline",
     {"pii": True, "types": ["name"]}),
]

FEW_SHOT_SETS = {"v1": FEW_SHOT, "v2": FEW_SHOT_V2}


def parse_prediction(raw: str):
    """Extract the first JSON object; return None when unparseable."""
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "pii" not in obj:
        return None
    types = obj.get("types") or []
    if not isinstance(types, list):
        types = []
    return {"pii": bool(obj["pii"]), "types": [t for t in types if t in TYPES]}


def build_prompt(tokenizer, text: str, few_shot: bool, shots=None):
    """Mistral requires alternating user/assistant turns and rejects a
    standalone system role, so instructions ride on the first user turn.
    Identical shape to the training data in generate_dataset.py."""
    messages = []
    if few_shot:
        for i, (ex_text, ex_label) in enumerate(shots if shots is not None else FEW_SHOT):
            content = f"{SYSTEM_PROMPT}\n\nLine: {ex_text}" if i == 0 else f"Line: {ex_text}"
            messages.append({"role": "user", "content": content})
            messages.append({
                "role": "assistant",
                "content": json.dumps(ex_label, separators=(",", ":")),
            })
        messages.append({"role": "user", "content": f"Line: {text}"})
    else:
        messages.append({"role": "user", "content": f"{SYSTEM_PROMPT}\n\nLine: {text}"})
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero-shot", "few-shot", "lora"], required=True)
    ap.add_argument("--model", default="mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--test-file", default="test_raw.jsonl",
                    help="ood_test.jsonl for the hand-written out-of-distribution set")
    ap.add_argument("--tag", default="", help="suffix for the results filenames")
    ap.add_argument("--few-shot-set", choices=["v1", "v2"], default="v1",
                    help="which example pool to prompt with (see FEW_SHOT_SETS)")
    args = ap.parse_args()

    kwargs = {"adapter_path": args.adapter_path} if args.adapter_path else {}
    model, tokenizer = load(args.model, **kwargs)
    sampler = make_sampler(temp=0.0)

    rows = [json.loads(l) for l in open(HERE / "data" / args.test_file)]
    if args.limit:
        rows = rows[: args.limit]

    tp = fp = fn = tn = 0
    invalid = 0
    per_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in TYPES}
    predictions = []
    start = time.time()

    for row in rows:
        prompt = build_prompt(tokenizer, row["text"], few_shot=(args.mode == "few-shot"),
                              shots=FEW_SHOT_SETS[args.few_shot_set])
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=64, sampler=sampler,
                       verbose=False)
        pred = parse_prediction(raw)
        if pred is None:
            invalid += 1
            pred = {"pii": False, "types": []}  # unparseable counts as "no detection"
        gold = row["label"]

        if gold["pii"] and pred["pii"]:
            tp += 1
        elif gold["pii"] and not pred["pii"]:
            fn += 1
        elif not gold["pii"] and pred["pii"]:
            fp += 1
        else:
            tn += 1

        for t in TYPES:
            in_gold, in_pred = t in gold["types"], t in pred["types"]
            if in_gold and in_pred:
                per_type[t]["tp"] += 1
            elif in_gold:
                per_type[t]["fn"] += 1
            elif in_pred:
                per_type[t]["fp"] += 1

        predictions.append({"text": row["text"], "gold": gold, "pred": pred, "raw": raw.strip()})

    elapsed = time.time() - start
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def type_f1(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        return round(2 * p * r / (p + r), 3) if p + r else 0.0

    results = {
        "mode": args.mode,
        "model": args.model,
        "adapter_path": args.adapter_path,
        "n": n,
        "accuracy": round((tp + tn) / n, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "valid_json_rate": round((n - invalid) / n, 3),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_type_f1": {t: type_f1(c) for t, c in per_type.items()},
        "seconds": round(elapsed, 1),
        "seconds_per_example": round(elapsed / n, 2),
    }

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    name = f"{args.mode}{args.tag}"
    (out / f"{name}.json").write_text(json.dumps(results, indent=2))
    (out / f"{name}_predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in predictions)
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
