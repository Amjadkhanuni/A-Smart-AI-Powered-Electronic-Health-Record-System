# src/medical_api.py

from serpapi import GoogleSearch

# Function to recommend medicine based on the query
def recommend_medicine(query):
    # Simple example of how medicine recommendation could work based on query
    if "headache" in query.lower():
        return "Recommended Medicine: Panadol (Paracetamol) for headache relief."
    elif "fever" in query.lower():
        return "Recommended Medicine: Ibuprofen for fever relief."
    elif "cough" in query.lower():
        return "Recommended Medicine: Dextromethorphan for cough relief."
    else:
        return "No specific medicine found for the given query."

# Function to find the nearest pharmacy store (based on query)
def find_nearby_store(query):
    # Simple example of how store finding could work based on the query (e.g., medicine name)
    if "Panadol" in query:
        return "Nearest store for Panadol: ABC Pharmacy, 123 Main Street."
    elif "Ibuprofen" in query:
        return "Nearest store for Ibuprofen: XYZ Pharmacy, 456 Elm Street."
    else:
        return "No store found for the given query."

# Function to fetch medical data from GME Plus or similar service (you can implement your logic here)
def get_gmeplus_data(query):
    # Here, you can implement the logic to fetch data from GME Plus API or another medical knowledge base.
    # For now, returning a placeholder message.
    return f"Medical data for {query} retrieved successfully."

# Function to get Google Search results using SerpAPI
def get_google_answer(query):
    # Initialize parameters for the API
    params = {
        "q": query,  # The query to search
        "api_key": "66f2ae2e110659e4645da6463cf8bb92c76794caabfbaaa813be45096c9a8cd8",  # Replace with your actual SerpAPI key
        "engine": "google",  # Using Google search engine from SerpAPI
    }

    # Call SerpAPI to get Google search results
    search = GoogleSearch(params)
    results = search.get_dict()

    try:
        # Extract the snippet from the organic results
        snippet = results["organic_results"][0]["snippet"]
        return f"📘 Google Search Result:\n\n{snippet}"
    except KeyError:
        return "⚠️ No relevant information found in search."