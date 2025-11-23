# src/rag.py
import faiss
import numpy as np
import torch
import gc
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.logger import init_log, add_log

# ---------------------------------
# 1️⃣ Load Models
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Models loaded successfully on:", device)

# ✅ Use BioBERT for retrieval embeddings (medical domain)
embed_model = SentenceTransformer("gsarti/biobert-nli")

# ✅ Use a stronger generator model for reasoning (Flan-T5-Large)
# If your system is slow, change to "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ✅ Convert to half precision if GPU is available
if torch.cuda.is_available():
    gen_model.half()
gen_model = gen_model.to(device)

print("[INFO] Models loaded successfully on:", device)

# ---------------------------------
# 2️⃣ Load Text Corpus
# ---------------------------------
try:
    with open("models/texts.txt", "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]
    print(f"[INFO] Loaded {len(docs)} documents for retrieval.")
except FileNotFoundError:
    print("❌ Error: models/texts.txt not found. Run embedding step first.")
    exit()

# ---------------------------------
# 3️⃣ Retrieve Similar Documents
# ---------------------------------
def retrieve_top_k(query, index_path="models/faiss.index", k=3):
    with torch.no_grad():
        q_emb = embed_model.encode([query], convert_to_numpy=True)
    """Retrieve top-k most similar docs from FAISS index."""
    index = faiss.read_index(index_path)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    retrieved = [docs[i] for i in I[0]]
    return retrieved, D[0]

# ---------------------------------
# 4️⃣ Generate Answer
# ---------------------------------
def generate_answer(query, retrieved_texts):
    """Generate factual answer using retrieved EHR context."""
    # Limit doc length to avoid truncation
    short_docs = [t[:600] for t in retrieved_texts]
    context = "\n\n".join(short_docs)

        # ✅ Improved prompt for better instruction following
    prompt = f"""
You are an expert radiology assistant.
Read the following clinical report carefully and answer the question in one short, factual medical sentence.
Do not copy full sentences from the report.
If the answer is not clearly mentioned, say exactly:
"Information not found in the provided records."

--- Clinical Report ---
{context}
-----------------------

Question: {query}

Answer (concise and factual):
"""
    try:
        with torch.no_grad():     # ✅ added
            inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        outputs = gen_model.generate(
            **inputs,
            max_length=100,
            num_beams=2,
            repetition_penalty=2.0,
            temperature=0.7,
            top_p=0.9,
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # Post-process to limit overly long answers or repetitions
        decoded = decoded.split(". ")[0].strip()
        decoded = " ".join(dict.fromkeys(decoded.split()))  # remove exact repeated words
        decoded = decoded.replace("normal normal", "normal")

        # ✅ Handle "None" or blank results
        if not decoded or decoded.lower() in ["none", "not found"]:
            decoded = "Not enough information in report."

        return decoded

    except Exception as e:
        print("❌ Generation error:", e)
        return "Error during generation."

# ---------------------------------
# 5️⃣ Main Test Run + Evaluation
# ---------------------------------

# Simple evaluation: token-level precision/recall/F1
def simple_eval(prediction, reference):
    """
    Compute basic token-level precision, recall, and F1 between prediction and reference.
    Returns a dict with 'precision', 'recall', 'f1'.
    """
    import re
    from collections import Counter

    pred_tokens = re.findall(r"\w+", (prediction or "").lower())
    ref_tokens = re.findall(r"\w+", (reference or "").lower())

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    common = 0
    for tok, cnt in pred_counts.items():
        common += min(cnt, ref_counts.get(tok, 0))

    precision = common / len(pred_tokens) if pred_tokens else 0.0
    recall = common / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

# Simple unigram BLEU-like score with brevity penalty
def bleu_score(prediction, reference):
    """
    Compute a simple unigram-precision-based BLEU-like score with brevity penalty.
    Returns a float between 0.0 and 1.0.
    """
    import math
    import re
    from collections import Counter

    pred_tokens = re.findall(r"\w+", (prediction or "").lower())
    ref_tokens = re.findall(r"\w+", (reference or "").lower())

    if not ref_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    overlap = 0
    for tok, cnt in pred_counts.items():
        overlap += min(cnt, ref_counts.get(tok, 0))

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0

    # Brevity penalty
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    if pred_len == 0:
        bp = 0.0
    elif pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / pred_len)

    score = bp * precision
    return round(score, 4)


if __name__ == "__main__":
    init_log()

    test_queries = [
        "What abnormality is seen in the chest X-ray?",
        "Is there any sign of pleural effusion?",
        "What is the diagnosis in this report?",
        "Is the heart size normal?",
        "What treatment was given for pneumonia?"
    ]

    reference_answers = [
        "Pulmonary abnormality is seen.",
        "Yes, pleural effusion is present.",
        "Diagnosis is pneumonia.",
        "The heart size is normal.",
        "Treatment given was antibiotics for pneumonia."
    ]

    for q, ref in zip(test_queries, reference_answers):
        print(f"\n========================\nQuestion: {q}\n")

        retrieved, scores = retrieve_top_k(q, k=3)
        generated_answer = generate_answer(q, retrieved)

        # Evaluation
        metrics = simple_eval(generated_answer, ref)
        f1 = metrics['f1']
        bleu = bleu_score(generated_answer, ref)

        print("[INFO] Final Answer:", generated_answer)
        print(f"\nReference: {ref}")
        print(f"\nF1 Score: {f1}, BLEU Score: {bleu}")

        # Log results
        add_log(
            query=q,
            retrieved_docs=retrieved,
            generated_answer=generated_answer,
            reference_answer=ref,
            score=f"F1={f1}, BLEU={bleu}"
        )
        torch.cuda.empty_cache()
        gc.collect()

print("[INFO] All queries processed and logged successfully!")