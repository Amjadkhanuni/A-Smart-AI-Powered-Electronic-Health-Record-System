import sys
import os
# Add src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import streamlit as st
import time
from src.medical_api import get_gmeplus_data, recommend_medicine, find_nearby_store
from src.rag import retrieve_top_k, generate_answer
# Attempt to import Google Search API function; provide a graceful fallback if the module is missing.
try:
    from src.medical_api import get_google_answer  # Importing the Google Search API function
except Exception:
    # Fallback function used when src.serp_api can't be imported.
    def get_google_answer(query):
        return "⚠️ Google Search API not available in this environment. Please install or provide src/serp_api.py to enable API-based search."

# ------------------- Streamlit Page Config -------------------
st.set_page_config(
    page_title="Intelligent EHR QA System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS Styling -------------------
st.markdown("""
    <style>
        /* General page styling */
        .main {
            background-color: #f8fafc;
            padding: 2rem;
        }
        h1 {
            color: #003366;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton button {
            background-color: #005b96;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
        }
        .stButton button:hover {
            background-color: #0074cc;
        }
        .response-box {
            background-color: #e6f2ff;
            border-left: 5px solid #0074cc;
            padding: 1rem;
            border-radius: 10px;
        }
        footer {
            text-align: center;
            color: #777;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- Sidebar -------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("**Choose Mode:**")
mode = st.sidebar.radio("Response Mode", ["Hybrid (Default)", "Dataset Only", "API Only"])

threshold = st.sidebar.slider("Dataset similarity threshold", 0.0, 1.0, 0.4)
k = st.sidebar.slider("Top K Documents", 1, 10, 3)
st.sidebar.info("Developed by Abrar Khan & Muhammad Ibrar — FYP Project")

# ------------------- Header -------------------
st.title("🧠 Intelligent EHR QA System")
st.markdown("""
### 🩺 Medical Question Answering using Semantic Similarity & RAG  
Ask any **patient-specific** or **general medical** question below.  
The system intelligently combines **EHR dataset retrieval** and **external API knowledge**.
""")

# ------------------- Input Section -------------------
query = st.text_input("💬 Enter your question:")

# Function: Wikipedia and Google API (fallback)
import re

def get_api_answer(query):
    try:
        keywords = re.findall(r"[A-Za-z]+", query)
        if not keywords:
            return "⚠️ Please enter a valid question."
        
        filtered = [w for w in keywords if len(w) > 3]
        topic = filtered[-1] if filtered else keywords[-1]
        topic = topic.strip().title().replace(" ", "_")

        # First try Wikipedia summary API
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "extract" in data and data["extract"]:
                return f"📘 Source: Wikipedia\n\n{data['extract']}"

        # If still nothing found, try Google Search API fallback
        return get_google_answer(query)  # Use Google Search API for detailed response

    except Exception as e:
        return f"API Error: {e}"

def add_log(query, retrieved_docs, generated_answer):
    """
    Safe logger for QA interactions: appends a JSON line to app/logs/answers.log.
    Fails silently to avoid breaking the main app if logging errors occur.
    """
    try:
        import json
        import os
        import datetime

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, "answers.log")

        # Keep retrieved docs concise to avoid huge log entries
        safe_docs = None
        if isinstance(retrieved_docs, (list, tuple)):
            safe_docs = [d[:1000] for d in retrieved_docs]  # truncate each doc
        else:
            safe_docs = str(retrieved_docs)[:1000]

        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "query": query,
            "retrieved_docs": safe_docs,
            "answer": generated_answer
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception:
        # Do not raise errors from logging; keep app resilient
        try:
            # best-effort minimal feedback in Streamlit if available
            st.warning("Logging failed (non-fatal).")
        except Exception:
            pass

# ------------------- Process Button -------------------
if st.button("🔍 Get Answer"):
    if not query.strip():
        st.warning("Please enter a question to proceed.")
    else:
        # 🌀 Show spinner during processing
        with st.spinner("🔍 Processing your question... please wait..."):
            # Retrieve from dataset
            retrieved, scores = retrieve_top_k(query, k=k)

            if mode != "API Only":
                st.markdown("### 📄 Retrieved Context (from EHR Dataset)")
                for i, doc in enumerate(retrieved):
                    st.write(f"**Doc {i+1} (score {scores[i]:.3f})**")
                    st.write(doc[:400] + "...")
                    st.divider()

            # Decision logic to use API or Dataset response
            use_api = (mode == "API Only") or (len(scores) == 0 or max(scores) < threshold)

            if use_api:
                st.markdown("### 🌐 External API Response")
                answer = get_api_answer(query)  # Fetch answer from API (Google or Wikipedia)
            else:
                st.markdown("### 💡 EHR-based Answer")
                answer = generate_answer(query, retrieved)  # Use EHR dataset-based response

            # 💬 Typing animation (like ChatGPT)
            placeholder = st.empty()
            typed_text = ""
            for char in answer:
                typed_text += char
                placeholder.markdown(f"<div class='response-box'>{typed_text}</div>", unsafe_allow_html=True)
                time.sleep(0.02)  # speed of typing (lower = faster)

        st.success("✅ Response generated successfully!")

        # ------------------ Optional: Log results -----------------
        # Add log for generated answers (you can log if needed)
        add_log(query=query, retrieved_docs=retrieved, generated_answer=answer)

# ------------------- Footer -------------------
st.markdown("""
<hr>
<footer>
Developed by <b>Abrar Khan & Muhammad Ibrar</b> | Department of Computer Science | Intelligent EHR QA using RAG (FYP)
</footer>
""", unsafe_allow_html=True)