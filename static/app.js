const btn = document.getElementById("btn");
const clearBtn = document.getElementById("clear");
const textEl = document.getElementById("text");

const out = document.getElementById("out");
const badge = document.getElementById("badge");
const emojiEl = document.getElementById("emoji");
const labelEl = document.getElementById("label");
const whyEl = document.getElementById("why");

const metaEl = document.getElementById("meta");
const probsEl = document.getElementById("probs");

const confText = document.getElementById("confText");
const confBar = document.getElementById("confBar");

function setBadge(label) {
  badge.classList.remove("badge--ok", "badge--review", "badge--block", "badge--error");
  if (label === "OK") badge.classList.add("badge--ok");
  else if (label === "REVIEW") badge.classList.add("badge--review");
  else if (label === "BLOCK") badge.classList.add("badge--block");
  else badge.classList.add("badge--error");
}

function showError(msg) {
  out.classList.remove("hidden");
  setBadge("ERROR");
  emojiEl.textContent = "⚠️";
  labelEl.textContent = "ERROR";
  whyEl.classList.add("hidden");
  metaEl.textContent = msg;

  confText.textContent = "";
  confBar.style.width = "0%";
  probsEl.textContent = "";
}

btn.onclick = async () => {
  const text = textEl.value.trim();
  if (!text) return showError("Bitte Text eingeben.");

  const res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ text })
  });

  const data = await res.json();
  if (!res.ok) return showError(data.error || "Unbekannter Fehler.");

  out.classList.remove("hidden");

  // Badge + Emoji
  setBadge(data.final_label);
  emojiEl.textContent = data.emoji;
  labelEl.textContent = data.final_label;

  // Warum REVIEW?
  if (data.gated_to_review) {
    whyEl.classList.remove("hidden");
    whyEl.textContent =
      `Warum REVIEW? Der Text wurde als OK erkannt, aber die Confidence (${data.confidence.toFixed(3)}) ` +
      `liegt unter dem Schwellenwert (${data.min_confidence}).`;
  } else {
    whyEl.classList.add("hidden");
    whyEl.textContent = "";
  }

  // Confidence Anzeige
  confText.textContent = data.confidence.toFixed(3);
  const pct = Math.max(0, Math.min(100, Math.round(data.confidence * 100)));
  confBar.style.width = pct + "%";

  // Meta
  metaEl.textContent = `raw=${data.raw_label} · min_conf=${data.min_confidence}`;

  // Probabilities
  const probs = data.probs;
  probsEl.textContent =
    Object.entries(probs)
      .map(([k,v]) => `${k}: ${v.toFixed(3)}`)
      .join("\n");
};

clearBtn.onclick = () => {
  textEl.value = "";
  out.classList.add("hidden");
};
