import faiss
import numpy as np

import faiss

def create_vector_index(embeddings):
    """
    Create a FAISS vector index from embeddings.

    Args:
        embeddings (list): List of embeddings.

    Returns:
        FAISS Index
    """
    if not embeddings or len(embeddings[0]) == 0:
        raise ValueError("No valid embeddings provided for indexing.")

    import faiss
    dimension = len(embeddings[0])  # Ensure valid embeddings
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_vector_index(index, query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return distances, indices