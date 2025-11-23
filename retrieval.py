import numpy as np
import faiss
import os

# Paths
emb_path = "models/embeddings.npy"
text_path = "models/texts.txt"
index_path = "models/faiss.index"

# --- Step 1: Check if embeddings exist ---
if not os.path.exists(emb_path):
    print("❌ Embedding file not found. Please run `python -m src.embed` first.")
    exit()

print("📦 Loading embeddings and texts...")

# --- Step 2: Load embeddings and text chunks ---
embeddings = np.load(emb_path)
with open(text_path, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f.readlines()]

d = embeddings.shape[1]  # embedding dimension
print(f"📏 Embedding dimension detected: {d}")

# --- Step 3: Create FAISS index ---
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# --- Step 4: Save index ---
os.makedirs("models", exist_ok=True)
faiss.write_index(index, index_path)

# --- Step 5: Verification printout ---
print("✅ FAISS index created successfully!")
print(f"✅ Total text chunks indexed: {len(texts)}")
print(f"✅ Index saved at: {index_path}")
print("🚀 Retrieval index ready to use.")