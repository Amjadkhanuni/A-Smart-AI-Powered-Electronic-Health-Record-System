# src/eval.py
import os
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method1
# simple token-F1 implemented manually (no sklearn tokenization needed)
from typing import Dict

# Import your RAG functions (adjust import if your rag.py location/names differ)
# rag.py should expose retrieve_top_k(query, index_path, k) and generate_answer(query, retrieved_texts)
from src.rag import retrieve_top_k, generate_answer

def simple_eval(generated: str, reference: str) -> Dict[str, float]:
    """
    Token-level simple precision, recall, f1.
    Lowercasing and splitting on whitespace.
    """
    gen_tokens = [t for t in str(generated).lower().split() if t.strip()]
    ref_tokens = [t for t in str(reference).lower().split() if t.strip()]
    gen_set = set(gen_tokens)
    ref_set = set(ref_tokens)
    tp = len(gen_set & ref_set)
    precision = tp / (len(gen_set) + 1e-8)
    recall = tp / (len(ref_set) + 1e-8)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}

from nltk.translate.bleu_score import SmoothingFunction

def bleu_score(generated: str, reference: str) -> float:
    """
    Compute BLEU score using NLTK with smoothing to avoid zero scores for short sentences.
    """
    try:
        gen_tokens = str(generated).split()
        ref_tokens = str(reference).split()
        smooth = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5), smoothing_function=smooth)
    except Exception:
        score = 0.0
    return float(score)

def evaluate_model(validation_csv: str, top_k: int = 3):
    # create results folder
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(validation_csv)
    results = []

    for i, row in df.iterrows():
        question = row['question']
        gold = row['gold_answer']

        print(f"\n--- Q{i+1} ---")
        print("Question:", question)
        # 1) retrieve top_k docs (calls your retrieval)
        try:
            retrieved_texts, scores = retrieve_top_k(question, k=top_k)
        except TypeError:
            # if your retrieve_top_k signature is different, call with only question
            retrieved_texts, scores = retrieve_top_k(question)

        print(f"Retrieved {len(retrieved_texts)} documents (showing first 1):")
        if len(retrieved_texts) > 0:
            print(retrieved_texts[0][:300], "...")
        # 2) generate answer using RAG pipeline
        generated = generate_answer(question, retrieved_texts)
        print("Generated:", generated)
        print("Gold:", gold)

        # 3) compute metrics
        token_metrics = simple_eval(generated, gold)
        bscore = bleu_score(generated, gold)

        result_row = {
            'question': question,
            'generated': generated,
            'gold_answer': gold,
            'precision': token_metrics['precision'],
            'recall': token_metrics['recall'],
            'f1': token_metrics['f1'],
            'bleu': bscore
        }
        results.append(result_row)

    res_df = pd.DataFrame(results)
    out_path = os.path.join("results", "evaluation_results.csv")
    res_df.to_csv(out_path, index=False)
    print("\nEvaluation complete. Results saved to:", out_path)
    return res_df

if __name__ == "__main__":
    # change path if your data folder is elsewhere
    val_csv = "data/validation_questions.csv"
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation file not found: {val_csv}")
    evaluate_model(val_csv, top_k=3)