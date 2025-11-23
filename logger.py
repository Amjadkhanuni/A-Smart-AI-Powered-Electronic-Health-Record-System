import csv
import os
from datetime import datetime

LOG_PATH = "logs/logs.csv"

def init_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "query", "retrieved_docs", "generated_answer", "reference_answer", "score"])
    print(f"Log initialized at {LOG_PATH}")

def add_log(query, retrieved_docs, generated_answer, reference_answer=None, score=None):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # store only first 100 chars of docs for brevity
    retrieved_str = " | ".join([doc[:100].replace("\n", " ") for doc in retrieved_docs])
    with open(LOG_PATH, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, query, retrieved_str, generated_answer, reference_answer, score])