import json, joblib
import numpy as np
from collections import Counter
from datasets import load_from_disk

pipe = joblib.load("model/pipeline.joblib")
meta = json.load(open("model/metadata.json", encoding="utf-8"))
labels = meta["labels"]
min_conf = float(meta.get("min_confidence_for_hard_decision", 0.55))

ds = load_from_disk("data/processed")["test"]
X = ds["text_clean"]

proba = pipe.predict_proba(X)
raw_idx = proba.argmax(axis=1)
raw = np.array([labels[i] for i in raw_idx])
conf = proba.max(axis=1)

final = raw.copy()
mask_gate = (raw != "REVIEW") & (conf < min_conf)
final[mask_gate] = "REVIEW"

print("Raw counts  :", Counter(raw))
print("Final counts:", Counter(final))

block_idx = np.where(raw == "BLOCK")[0]
print("Raw BLOCK predictions:", len(block_idx))
if len(block_idx) > 0:
    qs = np.quantile(conf[block_idx], [0, 0.1, 0.5, 0.9, 1.0])
    print("BLOCK confidence quantiles:", qs)
