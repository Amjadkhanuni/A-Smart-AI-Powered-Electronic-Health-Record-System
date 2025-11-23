import pandas as pd
import re
import os

# Function to clean individual text entries
def clean_text(s: str) -> str:
    if pd.isna(s): 
        return ""
    s = str(s)
    s = s.replace("\n", " ").strip()       # remove new lines and strip spaces
    s = re.sub(r"\s+", " ", s)             # collapse multiple spaces into one
    return s

# Main preprocessing function
def preprocess_reports(inpath, outpath):
    # 1️⃣ Read the input CSV
    df = pd.read_csv(inpath)

    # 2️⃣ Select relevant text columns based on your dataset
    text_cols = []
    for col in ['findings', 'impression', 'indication', 'comparison', 'Problems', 'MeSH']:
        if col in df.columns:
            text_cols.append(col)

    # 3️⃣ Fallback: if no text columns found, select all object type columns
    if not text_cols:
        text_cols = df.select_dtypes(include=['object']).columns.tolist()

    # 4️⃣ Combine all selected text columns into one 'combined_text' column
    df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1).apply(clean_text)

    # 5️⃣ Drop empty rows
    df = df[df['combined_text'].str.strip() != '']

    # 6️⃣ Create output directory if not exists
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # 7️⃣ Save the cleaned CSV
    df.to_csv(outpath, index=False)
    print("✅ Saved cleaned file:", outpath, "| Total rows:", len(df))

# Run when file is executed directly
if __name__ == "__main__":
    preprocess_reports(
        "K:/I-EHRs-Project/data/raw/indiana_reports.csv",
        "K:/I-EHRs-Project/data/cleaned/indiana_reports_cleaned.csv"
    )

import importlib
import importlib.util

spacy = None
HAS_SPACY = False
spec = importlib.util.find_spec("spacy")
if spec is not None:
    try:
        spacy = importlib.import_module("spacy")
        HAS_SPACY = True
    except Exception as e:
        spacy = None
        HAS_SPACY = False
        print("Warning: failed to import spacy even though it's installed:", e)
else:
    spacy = None
    HAS_SPACY = False
    print("Warning: spacy not available (not installed), skipping entity extraction.")

nlp = None
if HAS_SPACY:
    try:
        nlp = spacy.load("en_core_sci_sm")
    except Exception as e:
        print("Warning: could not load model 'en_core_sci_sm', trying fallback models:", e)
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception as e2:
            print("Warning: could not load fallback model 'en_core_web_sm':", e2)
            nlp = None

df = pd.read_csv("../data/cleaned/indiana_reports_cleaned.csv")

# Extract entities
def extract_entities(text):
    if nlp is None:
        return []  # spacy or model not available, return empty list
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

if 'combined_text' in df.columns:
    df['Entities'] = df['combined_text'].apply(extract_entities)
else:
    print("Warning: 'combined_text' column not found in the CSV; skipping entity extraction.")

df.to_csv("../data/cleaned/indiana_reports_with_entities.csv", index=False)
print("Entities added and saved (or skipped if spacy/model unavailable).")

