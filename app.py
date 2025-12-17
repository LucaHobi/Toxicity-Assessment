import json
import os
import re

import joblib
from flask import Flask, jsonify, render_template, request

MODEL_PATH = "model/pipeline.joblib"
META_PATH = "model/metadata.json"

# exakt wie in preprocess.py (wichtig: gleiche Cleaning-Logik)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("|LBR|", " ")
    t = URL_RE.sub("<URL>", t)
    t = USER_RE.sub("<USER>", t)
    t = WS_RE.sub(" ", t).strip()
    return t

def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

pipe = joblib.load(MODEL_PATH)
meta = load_metadata()

LABELS = meta["labels"]  # ["OK","REVIEW","BLOCK"]
EMOJI = meta.get("emoji", {"OK": "😀", "REVIEW": "😐", "BLOCK": "😡"})
MIN_CONF = float(meta.get("min_confidence_for_hard_decision", 0.52))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024  # 32KB Request Limit

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    text_clean = clean_text(text)

    if not text_clean:
        return jsonify({"error": "Bitte Text eingeben."}), 400

    # proba: [p(OK), p(REVIEW), p(BLOCK)]
    proba = pipe.predict_proba([text_clean])[0]
    probs = {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}

    raw_label = max(probs, key=probs.get)
    confidence = probs[raw_label]

    # Confidence-Gating: wenn unsicher -> REVIEW
    final_label = raw_label
    gated = False
    
    if raw_label == "OK" and confidence < MIN_CONF:
        final_label = "REVIEW"
        gated = True


    return jsonify({
        "text_clean": text_clean,
        "raw_label": raw_label,
        "final_label": final_label,
        "emoji": EMOJI.get(final_label, "❓"),
        "confidence": confidence,
        "min_confidence": MIN_CONF,
        "gated_to_review": gated,
        "probs": probs,
    })

if __name__ == "__main__":
    app.run(debug=True)
