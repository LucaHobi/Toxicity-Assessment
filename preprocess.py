import re
from datasets import load_dataset, DatasetDict, Features, Value, ClassLabel

DATASET = "philschmid/germeval18"
OUT_DIR = "data/processed"
SEED = 42
VAL_SIZE = 0.15

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")

LABEL3_MAP = {
    "OTHER": "OK",                 # 😀
    "PROFANITY": "REVIEW",         # 😐
    "INSULT": "REVIEW",            # 😐
    "ABUSE": "BLOCK",              # 😡
}

LABEL3_ID = {"OK": 0, "REVIEW": 1, "BLOCK": 2}


def clean_text(t: str) -> str:
    if t is None:
        return ""
    t = t.replace("|LBR|", " ")
    t = URL_RE.sub("<URL>", t)
    t = USER_RE.sub("<USER>", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def transform(example):
    multi = example["multi"]
    label3 = LABEL3_MAP[multi]
    return {
        "text_clean": clean_text(example["text"]),
        "label3": label3,
        "label3_id": LABEL3_ID[label3],
        # optional: original labels behalten (für spätere Analyse)
        "binary": example["binary"],
        "multi": multi,
    }


def print_dist(ds, name):
    # kein Text ausgeben (potenziell beleidigend)
    from collections import Counter
    c = Counter(ds["label3"])
    total = len(ds)
    print(f"\n{name} size: {total}")
    for k in ["OK", "REVIEW", "BLOCK"]:
        v = c.get(k, 0)
        print(f"  {k:6s}: {v:4d}  ({v/total:.1%})")


def main():
    raw = load_dataset(DATASET)  # splits: train/test
    train_raw = raw["train"]
    test_raw = raw["test"]

    # Transform anwenden
    train_t = train_raw.map(transform, remove_columns=train_raw.column_names)
    test_t = test_raw.map(transform, remove_columns=test_raw.column_names)

    # Leere Texte entfernen (falls Cleaning alles rauswirft)
    train_t = train_t.filter(lambda x: len(x["text_clean"]) > 0)
    test_t = test_t.filter(lambda x: len(x["text_clean"]) > 0)
    
    
    features = Features({
        "text_clean": Value("string"),
        "label3": Value("string"),
        "label3_id": ClassLabel(names=["OK", "REVIEW", "BLOCK"]),
        "binary": Value("string"),
        "multi": Value("string"),
    })

    train_t = train_t.cast(features)
    test_t = test_t.cast(features)

    # Train/Val split (stratifiziert nach label3)
    try:
        split = train_t.train_test_split(
            test_size=VAL_SIZE,
            seed=SEED,
            stratify_by_column="label3_id",
        )
        train_final = split["train"]
        val_final = split["test"]
    except TypeError:
        # Falls deine datasets-Version stratify_by_column nicht kennt:
        raise SystemExit(
            "Deine 'datasets'-Version unterstützt stratify_by_column nicht.\n"
            "Fix: uv add -U datasets  (oder: uv add scikit-learn und wir machen den Split manuell)"
        )

    processed = DatasetDict({
        "train": train_final,
        "val": val_final,
        "test": test_t,
    })

    # Verteilungen ausgeben
    print_dist(processed["train"], "TRAIN")
    print_dist(processed["val"], "VAL")
    print_dist(processed["test"], "TEST")

    # Lokal speichern (für spätere Schritte)
    processed.save_to_disk(OUT_DIR)
    print(f"\nSaved processed dataset to: {OUT_DIR}")


if __name__ == "__main__":
    main()
