import streamlit as st
import requests
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('API_KEY')
client = Groq(api_key=API_KEY)

st.title("Answering Bot (RAG)")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to chunk the text into smaller parts
def chunk_text(text, chunk_size=500):
    # Split text into sentences
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to retrieve relevant context using embeddings
def retrieve_relevant_information(text_data, user_question=None, model=None):
    if not text_data:
        return "No text data provided."
    
    # Chunk the text into smaller parts
    chunks = chunk_text(text_data, chunk_size=500)
    
    # Create embeddings for each chunk
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    
    if user_question:
        # Encode the user's question
        question_embedding = model.encode([user_question], convert_to_tensor=True)
        
        # Calculate the similarity between the question and each chunk
        similarities = cosine_similarity(question_embedding.cpu(), chunk_embeddings.cpu())
        most_relevant_index = np.argmax(similarities)
        relevant_chunk = chunks[most_relevant_index]
        return relevant_chunk
    else:
        # Return the whole text if no question is asked
        return text_data

uploaded_file = st.file_uploader("📤 Upload a PDF or Text file", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        file_text = extract_text_from_pdf(uploaded_file)
    else:
        file_text = uploaded_file.getvalue().decode("utf-8")
    
    if file_text:  
        st.subheader("Extracted Text:")
        st.write(file_text[:500] + " ...")  # Show the first 500 characters of the extracted text

        # Load pre-trained model for embeddings
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Get relevant context based on the extracted text
        retrieved_context = retrieve_relevant_information(file_text, model=model)

        # Input for user's question
        user_question = st.text_input("Ask a question about the text:")

        if user_question:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_question},
                        {"type": "text", "text": retrieved_context},
                    ]
                }
            ]

            try:
                # Send the question and context to Groq API
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=1000
                )
                # Display the response
                answer = response.choices[0].message.content
                st.subheader("📝 AI Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error getting response from Groq AI: {e}")
    else:
        st.error("Failed to extract text from the file. Please check the file format or content.")
