import streamlit as st
import PyPDF2
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# Initialize Sentence Transformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up the Streamlit page
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to create embeddings and save as a knowledge base
def create_knowledge_base(text, file_name="pdf_knowledge.pkl"):
    sentences = text.split("\n")
    embeddings = embedding_model.encode(sentences)
    
    # Save embeddings
    with open(file_name, "wb") as f:
        pickle.dump((sentences, embeddings), f)

# Function to retrieve the most relevant text chunk
def retrieve_information(query, file_name="pdf_knowledge.pkl"):
    if not os.path.exists(file_name):
        return "No PDF knowledge base found. Please upload a PDF first."

    # Load stored embeddings
    with open(file_name, "rb") as f:
        sentences, embeddings = pickle.load(f)

    # Compute similarity
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    best_match_idx = similarities.argmax()

    return sentences[best_match_idx]

# Sidebar - PDF Upload
st.sidebar.header("Upload a PDF for Chat")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    create_knowledge_base(pdf_text)
    st.sidebar.success("PDF content processed successfully!")

# Chat UI
st.markdown("<h2 style='text-align: center;'>📄 PDF Chatbot</h2>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")

# Chat Input Form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question about the uploaded PDF...")
    submit_button = st.form_submit_button(label="Send")

# Process input
if submit_button and user_input:
    response = retrieve_information(user_input)  # Retrieve answer from PDF knowledge base
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("PDF Chatbot", response))
    st.rerun()
