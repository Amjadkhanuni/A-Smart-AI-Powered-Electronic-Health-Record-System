# ==============================================
# File: embed.py
# Purpose: Convert cleaned EHR text into embeddings for retrieval (with chunking)
# ==============================================

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================================================
# 🧩 CONFIGURATION SECTION
# =========================================================

# Choose one of the models below
# "all-MiniLM-L6-v2" → Fast & general-purpose
# "pritamdeka/S-BioBert-snli-scinli" → Medical domain model (slower but more accurate)
MODEL_NAME = "all-MiniLM-L6-v2"

# Chunk size (number of words per chunk)
CHUNK_SIZE = 200


# =========================================================
# 🧠 HELPER FUNCTIONS
# =========================================================

def chunk_text(text, max_words=CHUNK_SIZE):
    """
    Break long medical reports into smaller chunks for better retrieval accuracy.
    """
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def build_embeddings(clean_csv, out_emb_path, out_text_path):
    """
    Load cleaned EHR data, chunk text, and create embeddings for each chunk.
    """
    # Step 1: Load the cleaned dataset
    print(f"Loading cleaned file: {clean_csv}")
    df = pd.read_csv(clean_csv)

    # Step 2: Verify column presence
    if 'combined_text' not in df.columns:
        raise KeyError("❌ Column 'combined_text' not found in CSV. Please run preprocessing first.")

    # Step 3: Combine all chunks
    texts = df['combined_text'].astype(str).tolist()
    all_chunks = []
    for t in texts:
        for chunk in chunk_text(t):
            all_chunks.append(chunk)
    print(f"✅ Total text chunks prepared: {len(all_chunks)}")

    # Step 4: Load SentenceTransformer model
    print(f"🧠 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Step 5: Encode all chunks into embeddings
    print("⚙️ Encoding text chunks into embeddings (this may take a few minutes)...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    # Step 6: Ensure output directory exists
    os.makedirs(os.path.dirname(out_emb_path), exist_ok=True)

    # Step 7: Save embeddings as .npy
    np.save(out_emb_path, embeddings)
    print(f"✅ Embeddings saved at: {out_emb_path}")

    # Step 8: Save chunked text
    with open(out_text_path, "w", encoding="utf-8") as f:
        for t in all_chunks:
            f.write(t.replace("\n", " ") + "\n")
    print(f"✅ Text chunks saved at: {out_text_path}")

    # Step 9: Summary info
    print("📊 Embeddings shape:", embeddings.shape)
    print("📦 Total chunks encoded:", len(all_chunks))
    print("🚀 Embedding generation complete!")


# =========================================================
# 🧪 RUN DIRECTLY
# =========================================================
if __name__ == "__main__":
    build_embeddings(
        "../data/cleaned/indiana_reports_cleaned.csv",  # Correct path & filename
        "../models/embeddings.npy",
        "../models/texts.txt"
    )