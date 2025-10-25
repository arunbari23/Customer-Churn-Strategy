# scripts/evaluate_strategies.py
import os
import re
import pandas as pd

# Try the cleaned file first; fall back to the raw file if it doesn't exist
CLEAN = "outputs/customer_retention_strategies_clean.csv"
RAW   = "outputs/customer_retention_strategies.csv"
INP   = CLEAN if os.path.exists(CLEAN) else RAW
OUTP  = "outputs/evaluated_retention_strategies.csv"

df = pd.read_csv(INP)

# ---------- Signals ----------
# Broader action cues (use roots so "apologize"/"apology" both match)
ACTION_TERMS = [
    r"apolog", r"offer", r"assist", r"support", r"reduce", r"waiv", r"credit",
    r"refund", r"replace", r"prioritiz", r"escalat", r"call\s?back", r"follow\s?up",
    r"notify", r"clarif", r"explain", r"expedit", r"improv", r"simplif", r"streamlin",
    r"train", r"monitor", r"compensat", r"remind", r"extend", r"adjust", r"resolve",
    r"contact", r"reach\s?out", r"schedule", r"callback"
]

# Empathy/acknowledgement cues
EMPATHY_TERMS = [
    r"sorry", r"apolog", r"understand", r"appreciat", r"thank", r"acknowledg",
    r"we\s+recognize", r"we\s+hear", r"regret", r"concern"
]

# Specificity cues (deadlines, numbers, SLAs, channels)
SPECIFICITY_PATTERNS = [
    r"\b\d+\s?(minutes?|hours?|days?|weeks?)\b",
    r"\bwithin\s+\d+\s?(minutes?|hours?|days?)\b",
    r"\b48[-\s]?hours?\b|\b24[-\s]?hours?\b|\b7[-\s]?days?\b",
    r"\bdeadline\b|\bSLA\b|\btimeframe\b",
    r"\bemail\b|\bchat\b|\bphone\b|\bportal\b|\bbranch\b"
]

# Vague phrases to penalize
VAGUE_TERMS = [
    r"we value your feedback", r"strive to", r"do our best", r"work on it",
    r"look into it", r"as soon as possible", r"soon"
]

# ---------- Scoring weights ----------
W_ACTION      = 2.0
W_EMPATHY     = 1.0
W_SPECIFICITY = 1.5
W_LEN_OK      = 1.0   # reward if >= 16 words
W_VAGUE_PEN   = -1.0  # penalty per vague phrase

# Sentiment-aware threshold (slightly lower for Negative to encourage empathy/action)
BASE_THRESHOLD       = 3.5
NEGATIVE_BONUS_DELTA = -0.3  # threshold becomes 3.2 for negatives

def count_matches(patterns, text):
    return sum(bool(re.search(p, text)) for p in patterns)

def score_strategy(row):
    s_raw = row.get("Retention Strategy", "")
    s = str(s_raw).strip().lower()

    if not s:
        return 0.0, "empty"

    # basic cleanup double spaces -> single
    s = re.sub(r"\s+", " ", s)

    n_action      = count_matches(ACTION_TERMS, s)
    n_empathy     = count_matches(EMPATHY_TERMS, s)
    n_specific    = count_matches(SPECIFICITY_PATTERNS, s)
    n_vague       = count_matches(VAGUE_TERMS, s)
    word_count    = len(re.findall(r"\w+", s))
    len_ok        = word_count >= 16

    score = (
        n_action * W_ACTION
        + n_empathy * W_EMPATHY
        + n_specific * W_SPECIFICITY
        + (W_LEN_OK if len_ok else 0.0)
        + (n_vague * W_VAGUE_PEN)
    )

    reason_parts = []
    if n_action:      reason_parts.append(f"action:{n_action}")
    if n_empathy:     reason_parts.append(f"empathy:{n_empathy}")
    if n_specific:    reason_parts.append(f"specific:{n_specific}")
    if len_ok:        reason_parts.append("len>=16")
    if n_vague:       reason_parts.append(f"vague:{n_vague}")

    reason = ", ".join(reason_parts) if reason_parts else "no_signals"

    # threshold (sentiment-aware)
    sent = str(row.get("Sentiment", "")).lower()
    thresh = BASE_THRESHOLD + (NEGATIVE_BONUS_DELTA if "negative" in sent else 0.0)

    label = "Effective" if score >= thresh and n_action >= 1 else "Needs Review"
    return score, reason, label

# Apply
scores, reasons, labels = [], [], []
for _, r in df.iterrows():
    sc, rsn, lbl = score_strategy(r)
    scores.append(sc); reasons.append(rsn); labels.append(lbl)

df["Score"] = scores
df["Reason"] = reasons
df["Strategy Effectiveness"] = labels

df.to_csv(OUTP, index=False)
print(f"✅ saved -> {OUTP}")

# This script evaluates the quality and effectiveness of customer retention strategies based on their textual content. 
# It starts by checking if a cleaned version of the data (customer_retention_strategies_clean.csv) exists; if not, it uses the raw version instead. 
# After loading the data into a DataFrame, the script defines several keyword-based signal lists that help identify useful patterns in each strategy:

#     1) Action terms (e.g., “offer”, “assist”, “refund”) indicate concrete actions being taken.

#     2) Empathy terms (e.g., “sorry”, “understand”, “thank”) show acknowledgment of customer feelings.

#     3) Specificity patterns look for measurable details like deadlines (“within 48 hours”) or communication channels (“email”, “phone”).

#     4) Vague terms (e.g., “we’ll look into it”, “as soon as possible”) are penalized because they suggest weak or unclear responses.

# Each strategy is then scored using weighted criteria: actions contribute the most points, followed by specificity, 
# empathy, and adequate length (16+ words). Vague language reduces the score. 
# These weights mimic how a human evaluator might judge the clarity and professionalism of a customer support response.

# The script also adjusts its evaluation based on sentiment — if a customer’s sentiment is negative, 
# the required threshold for labeling a strategy as “Effective” is slightly reduced, encouraging more empathetic or action-driven responses. 
# For every strategy, it records three things: the numeric score, 
# the reason breakdown (e.g., “action:2, empathy:1, len>=16”), and a final label (“Effective” or “Needs Review”).

# Finally, all these results are saved into a new CSV file (evaluated_retention_strategies.csv), 
# making it easy to analyze which strategies are performing well and which need improvement.

# In short, this script works like an automatic checker that reads each 
# customer retention message and looks for useful words or patterns — like actions, empathy, or clear details. 
# It then gives each message a score based on how strong and helpful it sounds, 
# and finally marks it as either “Effective” or “Needs Review” depending on how good it is.


# Score=(naction​×2.0)+(nempathy​×1.0)+(nspecific​×1.5)+(1.0 if len ≥ 16)−(nvague​×1.0)
# if score >= 3.5 effective 
# if score < 3.5 needs review 
