#!/usr/bin/env python3
"""Build the PII-detection dataset from permissively licensed public data.

Sources (both Apache-2.0, commercial use allowed, attribution in the article)
---------------------------------------------------------------------------
Positives : DataikuNLP/kiji-pii-training-data. Human-shaped multilingual PII
            corpus covering all six types we model, including IBAN and date
            of birth, with an author-provided train/test split we reuse so
            the test set cannot leak into training.
Negatives : witfoo/syslog-to-artifact. Real firewall and system syslog.

Two earlier candidates were rejected on purpose, and the article explains why:
  - ai4privacy/pii-masking-200k: dual license, free only for individuals and
    companies of three staff or fewer, so not safely usable for commercial
    content.
  - LogHub: licensed for "research or academic work" only, and several of its
    systems (BGL, Mac, Linux, OpenSSH) contain real usernames, home directory
    paths and email addresses. Labeling those lines "no PII" would have taught
    the model the exact opposite of the task.

Anti-shortcut design
--------------------
If every positive were prose and every negative a log line, the model would
separate the classes by writing style and never learn what PII is. So all
four quadrants exist: prose and log, on both sides of the label. Filler
vocabulary ("the customer") also appears in positives, otherwise it becomes a
giveaway for the negative class.

Output: data/train.jsonl, valid.jsonl, test.jsonl (mlx-lm chat format) and
data/test_raw.jsonl for the eval harness.
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

random.seed(42)
HERE = Path(__file__).parent
OUT = HERE / "data"
SPLITS = {"train": 8000, "valid": 800, "test": 800}
LOG_SHAPE_CAP = 40      # max lines sharing one log template, to avoid a
                        # single firewall message dominating the negatives

SYSTEM_PROMPT = (
    "You are a PII detector. Given one line of text, reply with strict JSON "
    'only: {"pii": true/false, "types": [...]}. Allowed types: "email", '
    '"phone", "name", "iban", "address", "dob". Use an empty list when '
    '"pii" is false. No other text.'
)

LANGS = {"English": "en", "French": "fr"}

LABEL_MAP = {
    "EMAIL": "email",
    "PHONENUMBER": "phone",
    "FIRSTNAME": "name", "SURNAME": "name",
    "IBAN": "iban",
    "STREET": "address", "BUILDINGNUM": "address", "CITY": "address",
    "ZIP": "address", "STATE": "address",
    "DATEOFBIRTH": "dob",
}

# Strong PII we do not model. Leaving it in a positive would mean unlabeled
# PII in the text; leaving it in a negative would teach that real PII is not
# PII. Dropping whole rows costs 75% of the corpus, so instead every span of
# these types is replaced by a role word before the row is used.
BLOCKED = {
    "SSN", "CREDITCARDNUMBER", "PASSWORD", "SECURITYTOKEN", "DRIVERLICENSENUM",
    "NATIONALID", "PASSPORTID", "TAXNUM", "LICENSEPLATENUM", "IDCARDNUM",
    "USERNAME", "RUT",
}

BLOCKED_WORDS = {
    "en": {
        "SSN": "the reference number", "CREDITCARDNUMBER": "the card on file",
        "PASSWORD": "the stored credential", "SECURITYTOKEN": "the API token",
        "DRIVERLICENSENUM": "the licence reference",
        "NATIONALID": "the national reference",
        "PASSPORTID": "the travel document", "TAXNUM": "the tax reference",
        "LICENSEPLATENUM": "the fleet vehicle", "IDCARDNUM": "the badge number",
        "USERNAME": "the login", "RUT": "the tax reference",
    },
    "fr": {
        "SSN": "le numero de reference", "CREDITCARDNUMBER": "la carte enregistree",
        "PASSWORD": "l'identifiant stocke", "SECURITYTOKEN": "le jeton d'API",
        "DRIVERLICENSENUM": "la reference du permis",
        "NATIONALID": "la reference nationale",
        "PASSPORTID": "le document de voyage", "TAXNUM": "la reference fiscale",
        "LICENSEPLATENUM": "le vehicule de flotte", "IDCARDNUM": "le numero de badge",
        "USERNAME": "l'identifiant", "RUT": "la reference fiscale",
    },
}

ROLE_WORDS = {
    "en": {
        "FIRSTNAME": "the customer", "SURNAME": "the account holder",
        "TITLE": "the contact", "EMAIL": "the support address",
        "PHONENUMBER": "the hotline", "IBAN": "the company account",
        "DATEOFBIRTH": "the enrolment date", "STREET": "the main office",
        "BUILDINGNUM": "the depot", "CITY": "our regional hub",
        "ZIP": "the delivery zone", "STATE": "the region",
        "COUNTRY": "the territory", "COMPANYNAME": "the vendor",
        "URL": "the internal wiki", "AGE": "the required age",
        "ORGANIZATION": "the department", "DOMAIN": "the internal domain",
    },
    "fr": {
        "FIRSTNAME": "le client", "SURNAME": "le titulaire du compte",
        "TITLE": "le contact", "EMAIL": "l'adresse du support",
        "PHONENUMBER": "la hotline", "IBAN": "le compte de la societe",
        "DATEOFBIRTH": "la date d'inscription", "STREET": "le siege",
        "BUILDINGNUM": "le depot", "CITY": "notre antenne regionale",
        "ZIP": "la zone de livraison", "STATE": "la region",
        "COUNTRY": "le territoire", "COMPANYNAME": "le prestataire",
        "URL": "le wiki interne", "AGE": "l'age requis",
        "ORGANIZATION": "le service", "DOMAIN": "le domaine interne",
    },
}

FIELD_NAME = {"email": "user", "phone": "callback", "name": "requester",
              "iban": "account", "address": "site", "dob": "birthdate"}

# Screens a "negative" that actually carries identity data. Deliberately
# wider than an email regex: the LogHub mistake was missing usernames in
# home directory paths.
CONTACT_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]{2,}"
    r"|\+\d[\d ().-]{7,}\d"
    r"|\b[A-Z]{2}\d{2}[A-Z0-9]{10,}\b"
    r"|/home/\w+|/Users/\w+"
    r"|\buser(name)?\s*[=:]\s*\w+"
    r"|\buid\s*[=:]\s*[a-z]{3,}",
    re.I,
)

# Syslog lines that carry account names in prose form ("Accepted password for
# johndoe"). Only ~4% of the corpus, so dropping them wholesale is cheaper
# than trying to parse them, and safer than mislabeling them as PII-free.
AUTH_RE = re.compile(
    r"sshd|sudo|\bsu\[|login|authentication|session opened"
    r"|\bfor [a-z][a-z0-9_.-]{2,}\b",
    re.I,
)

# Kiji's privacy_mask is not exhaustive: its coreferences field is empty on
# every row, and some later mentions of a person are simply not annotated
# ("Dubois a signe" after the annotated "Alice Dubois"). Substituting only
# the annotated spans therefore leaves real names inside would-be negatives.
# The fix is to screen every negative against a name vocabulary harvested
# from the whole corpus, and drop any candidate that still contains one.
NAME_VOCAB = set()


def has_leftover_name(text: str) -> bool:
    tokens = {t.strip(".,;:!?'\"()[]").lower() for t in text.split()}
    return bool(tokens & NAME_VOCAB)


def normalize(text: str) -> str:
    return " ".join(str(text).split())


def substitute(text: str, spans, replace_fn) -> str:
    """Rewrite spans right to left so earlier offsets stay valid."""
    out = text
    for s in sorted(spans, key=lambda s: s["start"], reverse=True):
        out = out[: s["start"]] + replace_fn(s) + out[s["end"]:]
    return out


def load_kiji():
    """Return per-split lists of usable rows: (text, spans, lang)."""
    ds = load_dataset("DataikuNLP/kiji-pii-training-data")
    usable = {}
    corrupted = 0
    for split in ("train", "test"):
        rows = []
        for row in ds[split]:
            lang = LANGS.get(row["language"])
            if not lang:
                continue
            text_raw = row["text"]
            if "[" in text_raw:         # residual template artifact
                continue
            # 4.7% of kiji's French rows are encoding-corrupted: accented
            # characters arrive as NUL bytes ("ete" written "\x00t\x00").
            # Training on them would teach the model mojibake French.
            if "\x00" in text_raw or "�" in text_raw:
                corrupted += 1
                continue
            spans = list(row["privacy_mask"])
            blocked = [s for s in spans if s["label"] in BLOCKED]
            if blocked:
                # neutralize out-of-scope PII, then re-derive the spans that
                # survive (offsets shift, so recompute by searching values)
                text = substitute(
                    row["text"], blocked,
                    lambda s: BLOCKED_WORDS[lang].get(s["label"], "the reference"))
                kept = []
                for s in spans:
                    if s["label"] in BLOCKED:
                        continue
                    idx = text.find(s["value"])
                    if idx == -1:
                        kept = None
                        break
                    kept.append({**s, "start": idx, "end": idx + len(s["value"])})
                if kept is None:
                    continue
                spans = kept
            else:
                text = row["text"]
            rows.append((text, spans, lang))
            for s in spans:
                if s["label"] in ("FIRSTNAME", "SURNAME"):
                    for tok in s["value"].split():
                        tok = tok.strip(".,;:'\"-").lower()
                        if len(tok) >= 3:
                            NAME_VOCAB.add(tok)
        usable[split] = rows
        print(f"  kiji {split}: {len(rows)} usable EN/FR rows")
    print(f"  dropped {corrupted} encoding-corrupted rows (NUL bytes for accents)")
    print(f"  name vocabulary for negative screening: {len(NAME_VOCAB)} tokens")
    return usable


def build_from_kiji(rows, n_pos, n_prose_neg, n_hybrid):
    """Positives, style-matched negatives and hybrid positives from one pool.

    Rows are handed to a shuffled list of slots rather than to a priority
    chain. With a chain, whichever bucket sits last (hybrids) starves once
    the pool runs out; with slots every bucket fills at the same rate.
    """
    random.shuffle(rows)
    slots = ["pos"] * n_pos + ["neg"] * n_prose_neg + ["hyb"] * n_hybrid
    random.shuffle(slots)
    positives, prose_negatives, hybrids, values = [], [], [], []
    slot_i = 0

    for text, spans, lang in rows:
        if slot_i >= len(slots):
            break
        want = slots[slot_i]
        mapped = [s for s in spans if s["label"] in LABEL_MAP]
        words = ROLE_WORDS[lang]

        if want == "pos":
            if not mapped:
                continue
            types = sorted({LABEL_MAP[s["label"]] for s in mapped})
            positives.append((normalize(text), {"pii": True, "types": types},
                              lang, "prose"))
            for s in mapped:
                if len(s["value"]) < 60:
                    values.append((LABEL_MAP[s["label"]], s["value"].strip()))

        elif want == "neg":
            # every PII span becomes a generic role word
            clean = normalize(substitute(
                text, spans,
                lambda s: words.get(s["label"], "the reference")))
            if CONTACT_RE.search(clean) or has_leftover_name(clean):
                continue    # keep the slot, try the next row
            prose_negatives.append((clean, {"pii": False, "types": []},
                                    lang, "prose"))

        else:  # hybrid: role words everywhere except one real value, so the
               # filler vocabulary is not a tell for the negative class
            if not mapped:
                continue
            keep = random.choice(mapped)
            kept_id = (keep["start"], keep["end"])
            clean = normalize(substitute(
                text, spans,
                lambda s: s["value"] if (s["start"], s["end"]) == kept_id
                else words.get(s["label"], "the reference")))
            hybrids.append((clean, {"pii": True,
                                    "types": [LABEL_MAP[keep["label"]]]},
                            lang, "prose"))

        slot_i += 1
    return positives, prose_negatives, hybrids, values


def load_logs(limit: int):
    """Real syslog, deduplicated, capped per template, screened for identity."""
    ds = load_dataset("witfoo/syslog-to-artifact", split="train", streaming=True)
    seen, per_shape, rows, dropped = set(), defaultdict(int), [], 0
    for row in ds:
        line = normalize(row["input_text"])
        if not (40 <= len(line) <= 400) or line in seen:
            continue
        seen.add(line)
        if CONTACT_RE.search(line) or AUTH_RE.search(line):
            dropped += 1
            continue
        shape = re.sub(r"\d+", "#", line)[:60]
        if per_shape[shape] >= LOG_SHAPE_CAP:
            continue
        per_shape[shape] += 1
        rows.append((line, {"pii": False, "types": []}, "en", "log"))
        if len(rows) >= limit:
            break
    print(f"  syslog: {len(rows)} kept across {len(per_shape)} templates, "
          f"{dropped} dropped by the identity screen")
    return rows


def make_log_positives(values, logs, count):
    """Real PII injected into real log lines, so log style also appears on
    the positive side of the label."""
    by_type = defaultdict(list)
    for t, v in values:
        by_type[t].append(v)
    types = [t for t in by_type if by_type[t]]
    out = []
    for _ in range(count):
        line = random.choice(logs)[0]
        t = random.choice(types)
        out.append((normalize(f"{line} {FIELD_NAME[t]}={random.choice(by_type[t])}"),
                    {"pii": True, "types": [t]}, "en", "log"))
    return out


def to_chat(text: str, label: dict) -> dict:
    # Mistral rejects a standalone system role and wants alternating turns.
    return {
        "messages": [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nLine: {text}"},
            {"role": "assistant", "content": json.dumps(label, separators=(",", ":"))},
        ]
    }


def assemble(kiji_rows, logs, size):
    """Half positives, half negatives, both styles on both sides."""
    n_prose_pos = int(size * 0.30)
    n_hybrid = int(size * 0.10)
    n_log_pos = int(size * 0.10)
    n_prose_neg = int(size * 0.25)
    n_log_neg = size - n_prose_pos - n_hybrid - n_log_pos - n_prose_neg

    pos, prose_neg, hybrids, values = build_from_kiji(
        kiji_rows, n_prose_pos, n_prose_neg, n_hybrid)
    log_pos = make_log_positives(values, logs, n_log_pos)
    print(f"    prose+ {len(pos)}/{n_prose_pos}, hybrid+ {len(hybrids)}/{n_hybrid}, "
          f"log+ {len(log_pos)}/{n_log_pos}, prose- {len(prose_neg)}/{n_prose_neg}, "
          f"log- {min(len(logs), n_log_neg)}/{n_log_neg}")
    rows = pos + hybrids + log_pos + prose_neg + logs[:n_log_neg]
    random.shuffle(rows)
    return rows, logs[n_log_neg:]


def main():
    OUT.mkdir(exist_ok=True)
    print("Loading kiji (Apache-2.0)...")
    kiji = load_kiji()
    print("Loading witfoo syslog (Apache-2.0)...")
    logs = load_logs(limit=14000)
    random.shuffle(logs)

    # test comes from kiji's own test split: no leakage by construction
    test_rows, logs = assemble(kiji["test"], logs, SPLITS["test"])
    trainval_rows, logs = assemble(kiji["train"], logs,
                                   SPLITS["train"] + SPLITS["valid"])

    # A shared log template can still surface the same line on both sides,
    # so drop any train/valid row whose text appears in the test set.
    test_texts = {t for t, _, _, _ in test_rows}
    before = len(trainval_rows)
    trainval_rows = [r for r in trainval_rows if r[0] not in test_texts]
    print(f"  removed {before - len(trainval_rows)} train rows duplicated in test")

    train_rows = trainval_rows[: SPLITS["train"]]
    valid_rows = trainval_rows[SPLITS["train"]: SPLITS["train"] + SPLITS["valid"]]

    for name, part in (("train", train_rows), ("valid", valid_rows),
                       ("test", test_rows)):
        with open(OUT / f"{name}.jsonl", "w") as fh:
            for text, label, _, _ in part:
                fh.write(json.dumps(to_chat(text, label)) + "\n")
    with open(OUT / "test_raw.jsonl", "w") as fh:
        for text, label, lang, style in test_rows:
            fh.write(json.dumps({"text": text, "label": label,
                                 "lang": lang, "style": style}) + "\n")

    stats = Counter()
    for _, label, lang, style in train_rows + valid_rows + test_rows:
        stats["positive" if label["pii"] else "negative"] += 1
        stats[f"{lang}/{style}"] += 1
        for t in label["types"]:
            stats[t] += 1
    print(f"\ntrain {len(train_rows)} / valid {len(valid_rows)} / test {len(test_rows)}")
    print(dict(stats))


if __name__ == "__main__":
    main()
