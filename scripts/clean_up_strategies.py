# file: scripts/clean_up_strategies.py
import re, pandas as pd

INP  = "outputs/customer_retention_strategies.csv"
OUTP = "outputs/customer_retention_strategies_clean.csv"

def clean(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.strip()

    # remove helper-y prefaces
    t = re.sub(r"(?i)^as a .*?assistant[:,]?\s*", "", t)
    t = re.sub(r"(?i)^based on the customer feedback[:,]?\s*", "", t)
    t = re.sub(r"(?i)^customer:\s*", "", t)
    t = re.sub(r"(?i)^sentiment:\s*.*?$", "", t)
    t = re.sub(r"(?i)^topic[s]?:\s*.*?$", "", t)
    t = re.sub(r"(?i)^action:\s*", "", t)

    # remove bullets/numbering and join lines
    t = re.sub(r"(?m)^\s*[-•\d]+\s*[).\-\:]\s*", "", t)
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # keep exactly two sentences
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts: return ""
    if len(parts) == 1: return parts[0]
    return (parts[0] + " " + parts[1]).strip()

df = pd.read_csv(INP)
df["Retention Strategy"] = df["Retention Strategy"].map(clean)
df.to_csv(OUTP, index=False)
print(f"✅ Cleaned file saved -> {OUTP}")


# This Python script (`clean_up_strategies.py`) is designed to clean and standardize a dataset of customer retention strategies stored in a CSV file.
#  It starts by importing the **re** (regular expressions) and **pandas** libraries, which are used for text processing and data handling, respectively. 
# The script reads the input CSV file (`customer_retention_strategies.csv`) into a DataFrame and focuses on cleaning the text in the `"Retention Strategy"` column using a custom `clean()` function.

# Inside this function, each text entry is first stripped of extra spaces and then processed with several **regex patterns** to remove unnecessary prefixes such as "As a ... assistant", "Customer:", "Sentiment:", or "Topics:" — these are likely helper-generated or metadata-like phrases. 
# It also removes any bullet points, numbering, or formatting symbols that may appear at the beginning of lines (like “1.”, “-”, “•”). 
# After that, it replaces line breaks with spaces and compresses multiple spaces into one, ensuring that the text is neatly formatted as a single line.

# Finally, the script limits each cleaned text to **only the first two sentences**, ensuring the result is concise and consistent. 
# The cleaned data is then saved as a new CSV file (`customer_retention_strategies_clean.csv`), and a confirmation message is printed.

# In short, this script systematically transforms messy, multi-line, 
# or AI-generated retention strategies into clean, human-readable, and uniformly formatted text suitable for further analysis or presentation.

