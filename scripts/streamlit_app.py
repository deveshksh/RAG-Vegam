import streamlit as st
from preprocess import preprocess_data
from embeddings import generate_embeddings
from vector_indexing import create_vector_index
from query_processing import process_query

# Load data
@st.cache_data  # Updated caching method
@st.cache_data
@st.cache_data
def load_data():
    # Define the data directory
    data_dir = "data/text"

    # Call preprocess_data with the data directory
    data = preprocess_data(data_dir)
    print("Preprocessed data:", data)  # Debugging

    if data.empty:
        raise ValueError("No data available for processing.")

    embeddings = generate_embeddings(data["content"].tolist())
    if not embeddings:
        raise ValueError("Embeddings generation failed. Check your input data or OpenAI API configuration.")

    vector_index = create_vector_index(embeddings)
    return data, vector_index



# Load data into the app
data, vector_index = load_data()

if data is not None and vector_index is not None:
    st.title("RAG System for MES in the Chemical Industry")

    query = st.text_input("Enter your query:")

    if query:
        response = process_query(query, vector_index, data.to_dict('records'))
        st.subheader("Response")
        st.write(response)