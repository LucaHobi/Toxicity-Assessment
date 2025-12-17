# Toxicity Triage (DE) – Uni-Demo (😀 OK / 😐 REVIEW / 😡 BLOCK)

Kleine ML-Web-App (Flask), die deutschen Text in drei Stufen klassifiziert:
- 😀 **OK** (nicht toxisch)
- 😐 **REVIEW** (grenzwertig, manuelle Prüfung)
- 😡 **BLOCK** (toxisch)

**Hinweis (Wichtig):** Das ist ein **Demo-Projekt**. Klassifikationen können falsch sein (Bias, False Positives/Negatives). Nicht als produktionsreifes Moderationssystem gedacht.

---

## Quickstart (Reproduzierbar mit `uv`)

### 1) Setup
```bash
uv sync
```

> Falls du kein Lockfile nutzt: Abhängigkeiten stehen im `pyproject.toml`. Du kannst auch direkt `uv run ...` verwenden und `uv` installiert on-demand.

### 2) Daten vorbereiten (Cleaning + Label-Mapping + Splits)

```bash
uv run python preprocess.py
```

Erzeugt: `data/processed/` mit Splits `train/val/test`.

### 3) Modell trainieren + evaluieren + speichern

```bash
uv run python train_model.py
```

Erzeugt:

* `model/pipeline.joblib`
* `model/metadata.json`

### 4) Web-App starten

```bash
uv run python app.py
```

Dann im Browser: `http://127.0.0.1:5000`

---

## Projekt-Scope (kurz)

**Ziel:** Eine einfache, nachvollziehbare Moderations-Triage als Web-Demo: Text rein → 😀/😐/😡 raus.
**Nicht-Ziel:** Vollständige Hate-Speech-Erkennung oder produktionsreifes Moderationssystem.

---

## Dataset

**Herkunft:** GermEval 2018 (deutsche Tweets) über Hugging Face Datasets: `philschmid/germeval18`
Genutzte Spalten:

* `text` (Tweet)
* `multi` (4 Klassen: OTHER / PROFANITY / INSULT / ABUSE)
* `binary` ist im Datensatz vorhanden, wird für dieses Projekt nicht zum Trainieren genutzt.

**Splits im Datensatz:** `train` (5009), `test` (3398)

---

## Label-Design (4 → 3 Klassen)

Wir trainieren **direkt 3 Klassen** (Scope A) durch Mapping der `multi` Labels:

| `multi` (Original) | Projektlabel | UI |
| ------------------ | ------------ | -- |
| OTHER              | OK           | 😀 |
| PROFANITY          | REVIEW       | 😐 |
| INSULT             | REVIEW       | 😐 |
| ABUSE              | BLOCK        | 😡 |

Warum so? Das entspricht einer Moderationslogik: “grenzwertig” bündelt mildere/gezielte Beleidigungen, “block” ist die härteste Kategorie.

---

## Preprocessing

In `preprocess.py` (und identisch in `app.py` für Inferenz):

* `|LBR|` → Leerzeichen (kommt häufig vor)
* URLs → `<URL>`
* `@mentions` → `<USER>`
* Whitespace normalisieren
* Leere Texte entfernen

**Train/Val Split:** aus dem ursprünglichen Train-Split wird ein Val-Split erzeugt (stratifiziert nach 3-Klassen-Label).

Resultierende Splits (nach Mapping):

* **TRAIN:** 4257 (OK 66.3%, REVIEW 13.3%, BLOCK 20.4%)
* **VAL:** 752 (OK 66.4%, REVIEW 13.3%, BLOCK 20.3%)
* **TEST:** 3398 (OK 66.2%, REVIEW 12.2%, BLOCK 21.7%)

Artefakt: `data/processed/` (lokal gespeichert via `save_to_disk()`)

---

## Modell

Baseline mit klassischer, schneller Pipeline (Scikit-learn):

**Gewähltes Modell:** `char_tfidf` + Logistic Regression

* Features: `TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_df=0.95, sublinear_tf=True)`
* Klassifikator: `LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")`

Warum char-n-grams? Robust gegenüber Schreibvarianten/Abkürzungen/Leetspeak und bei kurzen, informellen Texten oft stärker als word-n-grams.

Gespeichert als Pipeline: `model/pipeline.joblib`

---

## Evaluation (kurz)

Metrikfokus: **Macro-F1** (wegen Klassen-Ungleichgewicht; REVIEW ist relativ klein).

### Validation (bestes Modell)

* **Macro-F1:** 0.6285
* Confusion Matrix (rows=true, cols=pred):

  * `[[422,  32,  45],`
  * `[ 37,  46,  17],`
  * `[ 34,  29,  90]]`

### Test (bestes Modell, train+val refit)

* **Macro-F1:** 0.5491
* Confusion Matrix (rows=true, cols=pred):

  * `[[1934, 130, 184],`
  * `[ 198, 146,  69],`
  * `[ 376,  73, 288]]`

Beobachtung: **REVIEW** ist am schwierigsten (inhaltlich heterogen + weniger Beispiele).

---

## Inferenz-Policy (😀/😐/😡 Entscheidung)

Die App nutzt Modellwahrscheinlichkeiten `predict_proba()` und eine einfache Triage-Regel:

* `raw_label = argmax(probs)`
* **Confidence-Gating (Option A):**
  Wenn `raw_label == OK` und `confidence < 0.55` → setze `final_label = REVIEW`
  (BLOCK wird **nicht** “heruntergegated”, damit 😡 nicht verschwindet.)

Konfiguration steht in `model/metadata.json`:

* `min_confidence_for_hard_decision`

---

## Projektstruktur

```
toxicity-triage/
  app.py
  preprocess.py
  train_model.py
  data/processed/        # erzeugt durch preprocess.py
  model/
    pipeline.joblib      # erzeugt durch train_model.py
    metadata.json        # erzeugt durch train_model.py
  templates/index.html
  static/app.js
  static/styles.css
  pyproject.toml
```

---

## Privacy / Safety Notes (Demo)

* Keine Speicherung oder Ausgabe von problematischen Beispieltexten im README.
* Empfehlung: keine Rohtexte serverseitig loggen.
* Input wird nur für die aktuelle Prediction verarbeitet.

---

## Troubleshooting

**HF Hub Warning (unauthenticated requests):** optional ein `HF_TOKEN` setzen, um Limits zu erhöhen. Für dieses Projekt nicht zwingend.

---
