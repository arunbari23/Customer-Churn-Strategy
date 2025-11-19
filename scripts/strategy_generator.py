# scripts/gen_strategies_batched_robust.py
import os, json, re, time
import pandas as pd
from tqdm import tqdm
from scripts.ollama_utils import ollama_generate
from scripts.clean_up_strategies import clean as clean_strategy
from scripts.evaluate_strategies import score_strategy
# You will add regen_prompt() below

# ---------- Paths ----------
SENTIMENT_PATH = "data/feedback_sentiment.csv"
TOPICS_PATH    = "data/feedback_topics.csv"
OUTPUT_PATH    = "outputs/customer_retention_strategies.csv"
CHECKPOINT     = "outputs/_strategies_checkpoint.csv"

# ---------- Controls ----------
TOP_N       = 100
BATCH_SIZE  = 8
NUM_PREDICT = 170
TEMPERATURE = 0.45
CKPT_EVERY  = 5

os.makedirs("outputs", exist_ok=True)


# -------------------- Helpers --------------------
def clip(s, n=360):
    s = str(s).strip()
    return s if len(s) <= n else s[:n] + "..."


def clean_two_sentences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""
    t = re.sub(r"(?i)^(as a .*?assistant|based on .*?feedback)[:,]?\s*", "", t)
    t = re.sub(r"(?m)^\s*[-•\d]+\s*[).\-\:]\s*", "", t)
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()

    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else (parts[0] + " " + parts[1])


def build_lines(rows):
    lines = []
    for r in rows:
        topics_txt = ", ".join([t for t in r["topics"] if str(t).strip()][:5]) or "general"
        lines.append(f'{r["i"]}| {clip(r["feedback"], 360)} || {r["sentiment"]} || {topics_txt}')
    return "\n".join(lines)


# ------------------- REGEN PROMPT -------------------
def regen_prompt(feedback, sentiment, topics, idx):
    topics_txt = ", ".join(topics[:5]) or "general"
    return f"""
You previously generated a strategy for row {idx}, but it did not meet quality standards.
Generate a NEW two-sentence retention strategy.

Rules:
- exactly two sentences
- each sentence must have 14 words or fewer
- total output must not exceed 28 words
- empathetic + actionable
- no lists, bullet points, or assistant-style disclaimers

index| {feedback} || {sentiment} || {topics_txt}

Return ONLY the two-sentence strategy text.
""".strip()


# ------------------- ASK BATCH -------------------
def ask_batch(batch_rows):
    idxs = [r["i"] for r in batch_rows]
    lines = build_lines(batch_rows)

    prompt = (
        "You are a customer retention strategist.\n"
        "For each input line, return JSON ONLY in this exact schema:\n"
        '[{"index": <int>, "strategy": "<two short sentences>"}]\n'
        "Rules:\n"
        "- exactly two sentences\n"
        "- each sentence must have 14 words or fewer\n"
        "- total output must not exceed 28 words\n"
        "- empathetic, actionable sentences\n"
        "- no lists, headings, or self-reference.\n"
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
    for obj in data:
        idx = obj.get("index")
        strat = clean_two_sentences(obj.get("strategy", ""))
        if isinstance(idx, int) and strat:
            out[idx] = strat

    if any(i not in out for i in idxs):
        return None

    return out


# ------------------- CHECKPOINTS -------------------
def save_checkpoint(strategies):
    filled = [{"index": i, "strategy": s} for i, s in enumerate(strategies) if s]
    pd.DataFrame(filled).to_csv(CHECKPOINT, index=False)


def load_checkpoint(n_rows):
    strategies = [""] * n_rows
    if os.path.exists(CHECKPOINT):
        try:
            ck = pd.read_csv(CHECKPOINT)
            for _, r in ck.iterrows():
                strategies[int(r["index"])] = str(r["strategy"])
            print(f"↩️  Resumed from checkpoint with {sum(bool(s) for s in strategies)} rows.")
        except Exception:
            pass
    return strategies


# ----------------------- MAIN -----------------------
def main():
    sent = pd.read_csv(SENTIMENT_PATH)
    tops = pd.read_csv(TOPICS_PATH)

    if TOP_N:
        sent = sent.head(TOP_N)
        tops = tops.head(TOP_N)

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
    pbar.update(sum(bool(s) for s in strategies))

    k = 0
    batch_counter = 0

    try:
        while k < len(rows):

            slice_rows = rows[k:k + BATCH_SIZE]
            chunk = [r for r in slice_rows if not strategies[r["i"]]]

            if not chunk:
                k += BATCH_SIZE
                continue

            result = ask_batch(chunk)
            if result is None:
                if BATCH_SIZE > 4:
                    globals()["BATCH_SIZE"] = max(4, BATCH_SIZE // 2)
                else:
                    globals()["NUM_PREDICT"] = min(NUM_PREDICT + 20, 200)
                continue

            # ----------- CLOSED LOOP: regenerate until score >= 90 -----------
            for r in chunk:
                idx = r["i"]
                strat = result[idx]

                MAX_ATTEMPTS = 5
                attempt = 0
                final_strategy = None

                while attempt < MAX_ATTEMPTS:
                    attempt += 1

                    cleaned = clean_strategy(strat)

                    score_100, _, _ = score_strategy({
                        "Retention Strategy": cleaned,
                        "Sentiment": r["sentiment"]
                    })

                    if score_100 >= 90:
                        final_strategy = cleaned
                        break

                    # regenerate
                    strat = ollama_generate(
                        regen_prompt(r["feedback"], r["sentiment"], r["topics"], idx)
                    )

                if final_strategy is None:
                    final_strategy = cleaned

                strategies[idx] = final_strategy

            pbar.update(len(chunk))
            k += BATCH_SIZE
            batch_counter += 1

            if batch_counter % CKPT_EVERY == 0:
                save_checkpoint(strategies)

    except KeyboardInterrupt:
        print("\n⏸️ Interrupted — saving checkpoint...")
        save_checkpoint(strategies)
        raise

    finally:
        pbar.close()

    df_out = pd.DataFrame({
        "Customer Feedback": [r["feedback"] for r in rows],
        "Sentiment": [r["sentiment"] for r in rows],
        "Retention Strategy": strategies
    })
    df_out.to_csv(OUTPUT_PATH, index=False)

    if all(strategies):
        try:
            os.remove(CHECKPOINT)
        except:
            pass

    print(f"✅ Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
