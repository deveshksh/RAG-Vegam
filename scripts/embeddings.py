def generate_embeddings(text_list):
    """
    Generate embeddings for a list of texts using OpenAI's new API.

    Args:
        text_list (list): List of strings to generate embeddings for.

    Returns:
        list: List of embeddings.
    """
    embeddings = []
    for text in text_list:
        if not text.strip():  # Skip empty strings
            continue
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",  # Specify the model
                input=text
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for text: {text}\n{e}")
            # Skip appending None
    return [emb for emb in embeddings if emb is not None]  # Filter out any None values