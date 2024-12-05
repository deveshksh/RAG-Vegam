import openai

def process_query(query, vector_index, document_metadata, top_k=5):
    query_embedding = generate_embeddings([query])[0]
    distances, indices = search_vector_index(vector_index, query_embedding, top_k)
    context = [document_metadata[i]['content'] for i in indices[0]]
    prompt = f"Context: {' '.join(context)}\n\nQuery: {query}\n\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response['choices'][0]['text'].strip()