# Intelligent EHR Question Answering using Semantic Similarity & RAG

## Project Overview
This project implements a medical question-answering system using **Semantic Similarity** and **Retrieval-Augmented Generation (RAG)** on **Electronic Health Records (EHR)** data. The system intelligently combines **semantic embeddings** for retrieval and **T5-based generation** to provide accurate answers to medical queries.

### Key Features:
- **EHR Dataset Retrieval**: Retrieves medical data from **Electronic Health Records**.
- **External API Integration**: Uses **Google Search (SerpAPI)** and **Wikipedia** for answering medical questions when the dataset does not provide enough information.
- **Medicine Recommendations**: Recommends relevant medicines based on the user's query.
- **Nearest Store Locator**: Provides the nearest pharmacy or store that sells the recommended medicines.

## Folder Structure
- **data/**: Contains raw and cleaned datasets.
- **notebooks/**: Contains analysis notebooks.
- **src/**: Source code files for preprocessing, embedding, retrieval, and generation.
- **models/**: Trained models, embeddings, and indexes.
- **app/**: Simple demo interface.

## Setup Instructions
1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # For Windows