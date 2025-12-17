import json
import os
import joblib
from datasets import load_from_disk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATA_DIR = "data/processed"
MODEL_DIR = "model"
SEED = 42

LABELS = ["OK", "REVIEW", "BLOCK"]  # 0,1,2 wie in preprocess.py

def load_split(split_name: str):
    ds = load_from_disk(DATA_DIR)[split_name]
    texts = ds["text_clean"]
    y = ds["label3_id"]  # ClassLabel -> wird als int geliefert
    return texts, y

def make_pipelines():
    # 1) Word TF-IDF (klassisch)
    word = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    # 2) Char TF-IDF (oft stark bei Beleidigungen/Varianten/Leetspeak)
    char = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    return {"word_tfidf": word, "char_tfidf": char}

def eval_model(name, pipe, X_val, y_val):
    preds = pipe.predict(X_val)
    macro_f1 = f1_score(y_val, preds, average="macro")
    print(f"\n=== {name} ===")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, preds, target_names=LABELS, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, preds))
    return macro_f1

def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    candidates = make_pipelines()

    best_name, best_pipe, best_f1 = None, None, -1.0

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        f1 = eval_model(name, pipe, X_val, y_val)
        if f1 > best_f1:
            best_name, best_pipe, best_f1 = name, pipe, f1

    print(f"\nSelected model: {best_name} (val Macro-F1={best_f1:.4f})")

    # Final: Optional auf train+val neu fitten
    X_trainval = list(X_train) + list(X_val)
    y_trainval = list(y_train) + list(y_val)
    best_pipe.fit(X_trainval, y_trainval)

    # Test evaluation
    print("\n=== TEST evaluation (selected model) ===")
    test_preds = best_pipe.predict(X_test)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")
    print(classification_report(y_test, test_preds, target_names=LABELS, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, test_preds))

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "pipeline.joblib")
    joblib.dump(best_pipe, model_path)

    meta = {
        "labels": LABELS,
        "selected_candidate": best_name,
        "val_macro_f1": best_f1,
        "test_macro_f1": float(test_macro_f1),
        # UI-Emojis (für später)
        "emoji": {"OK": "😀", "REVIEW": "😐", "BLOCK": "😡"},
        # Optional: Confidence-gating defaults (kannst du später anpassen)
        "min_confidence_for_hard_decision": 0.55,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metadata to: {os.path.join(MODEL_DIR, 'metadata.json')}")

if __name__ == "__main__":
    main()
