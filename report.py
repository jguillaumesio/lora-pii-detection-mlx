#!/usr/bin/env python3
"""Assemble the three-way comparison tables from results/*.json.

Prints markdown ready to paste into the article, and writes
results/summary.md.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
MODES = ["zero-shot", "few-shot", "lora"]
LABELS = {
    "zero-shot": "Base, zero-shot",
    "few-shot": "Base, few-shot (6)",
    "lora": "LoRA fine-tuned",
}
TYPES = ["email", "phone", "name", "iban", "address", "dob"]


def load(mode):
    path = HERE / "results" / f"{mode}.json"
    return json.loads(path.read_text()) if path.exists() else None


def pct(x):
    return f"{round(x * 100)}%"


def main():
    res = {m: load(m) for m in MODES}
    available = [m for m in MODES if res[m]]

    lines = ["## Results", "", "| Metric | " + " | ".join(LABELS[m] for m in available) + " |"]
    lines.append("|---|" + "---|" * len(available))

    rows = [
        ("Accuracy", lambda r: pct(r["accuracy"])),
        ("Precision", lambda r: f"{r['precision']:.3f}"),
        ("Recall", lambda r: f"{r['recall']:.3f}"),
        ("F1", lambda r: f"{r['f1']:.3f}"),
        ("False positives (hard negatives)", lambda r: str(r["confusion"]["fp"])),
        ("Missed PII lines", lambda r: str(r["confusion"]["fn"])),
        ("Valid JSON", lambda r: pct(r["valid_json_rate"])),
        ("Seconds per line", lambda r: f"{r['seconds_per_example']:.2f}"),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(res[m]) for m in available) + " |")

    lines += ["", "## Per-type F1", "",
              "| PII type | " + " | ".join(LABELS[m] for m in available) + " |",
              "|---|" + "---|" * len(available)]
    for t in TYPES:
        lines.append(f"| {t} | " + " | ".join(f"{res[m]['per_type_f1'][t]:.3f}" for m in available) + " |")

    # Training facts, parsed from training.log when present.
    log = HERE / "training.log"
    if log.exists():
        text = log.read_text()
        peak = re.findall(r"Peak mem[a-z ]*([\d.]+) GB", text)
        val_losses = re.findall(r"Iter (\d+): Val loss ([\d.]+)", text)
        final = re.findall(r"Iter (\d+): Train loss ([\d.]+).*?Tokens/sec ([\d.]+)", text)
        lines += ["", "## Training", ""]
        if val_losses:
            best_iter, best = min(val_losses, key=lambda p: float(p[1]))
            lines.append(f"- Validation loss: {val_losses[0][1]} at iter {val_losses[0][0]}, "
                         f"best {best} at iter {best_iter}, last {val_losses[-1][1]} "
                         f"at iter {val_losses[-1][0]}")
        if peak:
            lines.append(f"- Peak memory: {max(float(p) for p in peak)} GB")
        if final:
            lines.append(f"- Final train loss: {final[-1][1]}")
        adapter = HERE / "adapters" / "adapters.safetensors"
        if adapter.exists():
            lines.append(f"- Adapter size: {adapter.stat().st_size / 1e6:.1f} MB")

    out = "\n".join(lines)
    (HERE / "results" / "summary.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
