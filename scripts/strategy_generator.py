# scripts/gen_strategies_batched_robust.py
import os, json, re, time
import pandas as pd
from tqdm import tqdm
from scripts.ollama_utils import ollama_generate  # MODEL = "qwen2.5:3b-instruct"

# ---------- Paths ----------
SENTIMENT_PATH = "data/feedback_sentiment.csv"
TOPICS_PATH    = "data/feedback_topics.csv"
OUTPUT_PATH    = "outputs/customer_retention_strategies.csv"
CHECKPOINT     = "outputs/_strategies_checkpoint.csv"

# ---------- Controls ----------
TOP_N       = 100        # <-- set to None to process ALL rows
BATCH_SIZE  = 8          # auto-reduces on failure
NUM_PREDICT = 140        # room for multiple 2-sentence outputs per batch
TEMPERATURE = 0.45
CKPT_EVERY  = 5          # save checkpoint every N batches

os.makedirs("outputs", exist_ok=True)

def clip(s, n=360):
    s = str(s).strip()
    return s if len(s) <= n else s[:n] + "..."

def clean_two_sentences(text: str) -> str:
    """Return up to two clean sentences. Safe for empty/None inputs."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""
    t = re.sub(r"(?i)^(as a .*?assistant|based on .*?feedback)[:,]?\s*", "", t)
    t = re.sub(r"(?m)^\s*[-•\d]+\s*[).\-\:]\s*", "", t)
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else (parts[0] + " " + parts[1])

def build_lines(rows):
    """index| <feedback> || <sentiment> || <topics>"""
    lines = []
    for r in rows:
        topics_txt = ", ".join([t for t in r["topics"] if str(t).strip()][:5]) or "general"
        lines.append(f'{r["i"]}| {clip(r["feedback"], 360)} || {r["sentiment"]} || {topics_txt}')
    return "\n".join(lines)

def ask_batch(batch_rows):
    idxs = [r["i"] for r in batch_rows]
    lines = build_lines(batch_rows)
    prompt = (
        "You are a customer retention strategist.\n"
        "For each input line, return JSON ONLY in this exact schema:\n"
        '[{\"index\": <int>, \"strategy\": \"<two short sentences>\"}]\n'
        "Rules: exactly two short, empathetic, actionable sentences. No lists, headings, or self-reference.\n"
        "INPUT FORMAT: index| <feedback> || <sentiment> || <topics>\n\n"
        f"{lines}\n\n"
        f"Return JSON for ALL these indexes: {idxs}. Return ONLY JSON."
    )
    resp = ollama_generate(prompt, num_predict=NUM_PREDICT, temperature=TEMPERATURE)
    start, end = resp.find("["), resp.rfind("]") + 1
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(resp[start:end])
    except Exception:
        return None

    out = {}
    if isinstance(data, list):
        for obj in data:
            idx = obj.get("index")
            strat = clean_two_sentences(obj.get("strategy", ""))
            if isinstance(idx, int) and strat:
                out[idx] = strat

    # require all indexes present
    if any(i not in out for i in idxs):
        return None
    return out

def save_checkpoint(strategies):
    filled = [{"index": i, "strategy": s} for i, s in enumerate(strategies) if s]
    pd.DataFrame(filled).to_csv(CHECKPOINT, index=False)

def load_checkpoint(n_rows):
    strategies = [""] * n_rows
    if os.path.exists(CHECKPOINT):
        try:
            ck = pd.read_csv(CHECKPOINT)
            for _, r in ck.iterrows():
                i = int(r["index"])
                if 0 <= i < n_rows:
                    strategies[i] = str(r["strategy"])
            print(f"↩️  Resumed from checkpoint with {sum(bool(s) for s in strategies)} filled rows.")
        except Exception:
            pass
    return strategies

def main():
    sent = pd.read_csv(SENTIMENT_PATH)
    tops = pd.read_csv(TOPICS_PATH)

    if TOP_N:
        sent = sent.head(TOP_N)
        tops = tops.head(TOP_N)

    # prepare rows
    rows = []
    for i in range(len(sent)):
        topics = tops.iloc[i].dropna().tolist() if i < len(tops) else []
        rows.append({
            "i": i,
            "feedback": sent.loc[i, "Customer Feedback"],
            "sentiment": sent.loc[i, "Sentiment"],
            "topics": topics
        })

    strategies = load_checkpoint(len(rows))

    pbar = tqdm(total=len(rows), desc="Generating (batched, robust)")
    filled_prev = sum(bool(s) for s in strategies)
    pbar.update(filled_prev)

    k = 0
    batch_counter = 0

    try:
        while k < len(rows):
            # current slice
            slice_rows = rows[k:k + BATCH_SIZE]
            # skip if already filled
            chunk = [r for r in slice_rows if not strategies[r["i"]]]
            if not chunk:
                k += BATCH_SIZE
                continue

            result = ask_batch(chunk)
            if result is None:
                # backoff: shrink batch; if already small, increase tokens slightly
                if BATCH_SIZE > 4:
                    globals()["BATCH_SIZE"] = max(4, BATCH_SIZE // 2)
                else:
                    globals()["NUM_PREDICT"] = min(NUM_PREDICT + 20, 200)
                continue

            for idx, strat in result.items():
                strategies[idx] = strat
            pbar.update(len(chunk))
            k += BATCH_SIZE

            batch_counter += 1
            if batch_counter % CKPT_EVERY == 0:
                save_checkpoint(strategies)

    except KeyboardInterrupt:
        print("\n⏸️  Interrupted — saving checkpoint...")
        save_checkpoint(strategies)
        raise
    finally:
        pbar.close()

    # final save
    df_out = pd.DataFrame({
        "Customer Feedback": [r["feedback"] for r in rows],
        "Sentiment": [r["sentiment"] for r in rows],
        "Retention Strategy": strategies
    })
    df_out.to_csv(OUTPUT_PATH, index=False)
    # remove checkpoint if complete
    if all(strategies):
        try: os.remove(CHECKPOINT)
        except Exception: pass
    print(f"✅ Saved -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

# this code takes two input files:

# feedback_sentiment.csv → contains customer feedback and its sentiment (positive, neutral, or negative).
# feedback_topics.csv → contains the key topics or categories related to each feedback (like delivery, billing, support, etc.).

# Then it sends both pieces of information (feedback + sentiment + topics) to a language model in batches, 
# asking it to generate a short, two-sentence customer retention strategy for each row — something actionable and empathetic.

# Finally, it combines all this into one output CSV file called customer_retention_strategies.csv, which has these columns:
# Customer Feedback
# Sentiment
# Retention Strategy (the generated response)

# In short: it automates the process of writing personalized retention strategies based on customer feedback and saves everything neatly into one CSV.
