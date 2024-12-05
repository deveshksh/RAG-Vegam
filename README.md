# Retrieval-Augmented Generation (RAG) System for MES

This project demonstrates a RAG system for handling queries on multiple data sources.

# High-Level Architecture Diagram

## 1. Overview
This architecture outlines a system for document retrieval, preprocessing, embedding, and interaction with a Language Model (LLM) for contextual query processing and response generation.

---

## 2. Diagram Representation

### Diagram Key Components:
1. **Data Sources**: 
   - Text files, PDFs, images, and other unstructured data sources.
2. **Preprocessing and Embedding Layers**:
   - Data cleaning and tokenization.
   - Embedding generation using models (e.g., OpenAI embeddings).
3. **Vector Database**:
   - Storage of embeddings in FAISS or another vector search engine.
4. **LLM Interaction**:
   - Contextual retrieval of data from the vector database.
   - Interaction with the LLM for processing.
5. **Query Processing and Response Generation**:
   - User queries flow into the system, interact with the vector database, and responses are generated via the LLM.

### Architecture Flow:
```plaintext
  Data Ingestion (Text, PDFs, Images) 
            ↓
   Preprocessing (Cleaning, Tokenization)
            ↓
   Embedding Generation (Text Embeddings)
            ↓
   Vector Database (e.g., FAISS)
            ↓
   Context Retrieval (Nearest Neighbor Search)
            ↓
   LLM Processing (Query + Context)
            ↓
   Response Generation (User Query Answer)


## Features
- Supports text, audio, video, and image data.
- Uses OpenAI embeddings and FAISS for vector search.
- Built with Streamlit for an interactive frontend.

## Setup
1. Clone the repository:
2. Install dependencies:
pip install -r requirements.txt
3. Run the application:
streamlit run scripts/streamlit_app.py
## Data Structure
- Store your data files in the `data/` folder.
- Organized into subfolders: `text/`, `audio/`, `video/`, `images/`.

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`.

## Usage
1. Place your data in the `data/` folder.
2. Run the Streamlit app and input a query.

## License
MIT