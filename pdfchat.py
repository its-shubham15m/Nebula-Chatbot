import streamlit as st
import PyPDF2
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import text_to_word_sequence

nltk.download('punkt')

# Set Streamlit page
st.set_page_config(
    page_title="Nebula - PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# ---------------------- THEME TOGGLE ----------------------

if "themes" not in st.session_state:
    st.session_state.themes = {
        "current_theme": "light",
        "refreshed": True,
        "light": {
            "theme.base": "dark",
            "theme.backgroundColor": "black",
            "theme.primaryColor": "#57B4BA",
            "theme.secondaryBackgroundColor": "#092635",
            "theme.textColor": "white",
            "button_face": "🌜"
        },
        "dark": {
            "theme.base": "light",
            "theme.backgroundColor": "white",
            "theme.primaryColor": "#57B4BA",
            "theme.secondaryBackgroundColor": "#F0EBE3",
            "theme.textColor": "#0a1464",
            "button_face": "🌞"
        }
    }

def change_theme():
    previous_theme = st.session_state.themes["current_theme"]
    tdict = st.session_state.themes["light"] if previous_theme == "light" else st.session_state.themes["dark"]
    
    for vkey, vval in tdict.items():
        if vkey.startswith("theme"):
            st._config.set_option(vkey, vval)
    
    st.session_state.themes["refreshed"] = False
    st.session_state.themes["current_theme"] = "dark" if previous_theme == "light" else "light"

if st.session_state.themes["refreshed"]:
    change_theme()

# Custom CSS for WhatsApp-style chat
st.markdown("""
    <style>
    .stForm {
        padding: 0;
        margin: 0;
        background-color: transparent;
    }
    .stTextInput > div > div > input {
        border: none;
        border-radius: 10px;
        padding: 8px 12px;
        background-color: #f0f0f0;
        color: #333;
        width: 100%;
    }
    .stButton > button {
        border: none;
        border-radius: 10px;
        padding: 8px 12px;
        background-color: #57B4BA;
        color: white;
        margin-left: 10px;
    }
    .stButton > button:hover {
        background-color: #468C91;
    }
    .chat-bubble {
        max-width: 70%;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        display: inline-block;
    }
    [data-baseweb="input"] input::placeholder {
        color: #888;
    }
    body[theme-mode="dark"] [data-baseweb="input"] input::placeholder {
        color: #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to preprocess PDF text
def preprocess_pdf_text(pdf_text):
    sentences = sent_tokenize(pdf_text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
    return sentences

# Function to create n-grams (N=1 to 6)
def create_ngram_contexts(sentences, min_n=1, max_n=6):
    contexts = []
    for n in range(min_n, max_n + 1):
        for i in range(len(sentences) - n + 1):
            context = " ".join(sentences[i:i + n])
            contexts.append((context, i, i + n - 1))  # Store context with start/end indices
    return contexts

# Function to process PDF and prepare TF-IDF
def process_pdf_contexts(sentences):
    contexts = create_ngram_contexts(sentences, min_n=1, max_n=6)
    context_texts = [context[0] for context in contexts]  # Extract just the text for TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(context_texts)
    return vectorizer, tfidf_matrix, contexts, sentences

# Function to get context-aware response
def get_context_aware_response(query, vectorizer, tfidf_matrix, contexts, sentences):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    best_context_idx = np.argmax(similarities)
    
    best_context, start_idx, end_idx = contexts[best_context_idx]
    context_sentences = best_context.split(". ")
    
    # Find the most relevant sentence within the context
    query_words = set(text_to_word_sequence(query.lower()))
    best_sentence = None
    max_overlap = 0
    
    for sentence in context_sentences:
        sentence_words = set(text_to_word_sequence(sentence.lower()))
        overlap = len(query_words.intersection(sentence_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sentence = sentence
    
    if best_sentence and max_overlap > 0:
        return best_sentence
    elif context_sentences:
        return context_sentences[0]  # Fallback to first sentence in context
    return "I couldn't find relevant information in the PDF. Please try rephrasing your query."

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a Page", ["PDF Chatbot", "Chatbot"])

if page == "Chatbot":
    os.system("streamlit run chatbot.py")
    st.stop()

# Theme Toggle Button
current_theme = st.session_state.themes["current_theme"]
if st.sidebar.button(
    st.session_state.themes[current_theme]["button_face"] + " Toggle Theme",
    key="theme_toggle"
):
    change_theme()
    st.rerun()

# Sidebar - Chatbot Details
st.sidebar.header("📌 Chatbot Info")
st.sidebar.text("🤖 Name: Nebula-AI")
st.sidebar.text("👨‍💻 Developer: Shubham Gupta")
st.sidebar.text("📜 Purpose: Context-Aware PDF Chatbot using N-Grams")

# Sidebar - PDF Upload
st.sidebar.header("Upload a PDF for Chat")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Initialize session state
if "pdf_vectorizer" not in st.session_state:
    st.session_state.pdf_vectorizer = None
if "pdf_tfidf_matrix" not in st.session_state:
    st.session_state.pdf_tfidf_matrix = None
if "pdf_contexts" not in st.session_state:
    st.session_state.pdf_contexts = None
if "pdf_sentences" not in st.session_state:
    st.session_state.pdf_sentences = None

# Process PDF upload
if uploaded_file and st.session_state.pdf_vectorizer is None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    sentences = preprocess_pdf_text(pdf_text)
    
    with st.spinner("Processing PDF with n-grams... This may take a moment."):
        vectorizer, tfidf_matrix, contexts, sentences = process_pdf_contexts(sentences)
        st.session_state.pdf_vectorizer = vectorizer
        st.session_state.pdf_tfidf_matrix = tfidf_matrix
        st.session_state.pdf_contexts = contexts
        st.session_state.pdf_sentences = sentences
    st.sidebar.success("PDF processed with n-grams in session!")

# Chat UI
st.markdown("<h2 style='text-align: center;'>📄 Nebula - PDF Chatbot</h2>", unsafe_allow_html=True)

user_bubble_color = "#57B4BA"
bot_bubble_color = st.session_state.themes[current_theme]["theme.secondaryBackgroundColor"]
text_color = st.session_state.themes[current_theme]["theme.textColor"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for sender, message in st.session_state.chat_history:
    if sender == "Nebula":
        st.markdown(
            f"""
            <div style='display: flex; align-items: center;'>
                <img src='https://img.icons8.com/fluency/48/000000/chatbot.png' width='40' height='40' style='margin-right:10px;'>
                <div class='chat-bubble' style='background-color: {bot_bubble_color}; color: {text_color};'>
                    <b>Nebula:</b> {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; justify-content: flex-end;'>
                <div class='chat-bubble' style='background-color: {user_bubble_color}; color: {text_color};'>
                    <b>You:</b> {message}
                </div>
                <img src='https://img.icons8.com/fluency/48/000000/user-male-circle.png' width='40' height='40' style='margin-left:10px;'>
            </div>
            """,
            unsafe_allow_html=True
        )

# Input form
with st.form(key='chat_form', clear_on_submit=True):
    cols = st.columns([4, 1])
    with cols[0]:
        user_input = st.text_input(
            "Type your message...", 
            key="user_input", 
            label_visibility="collapsed", 
            placeholder="Nebula reading PDF 📑... Which part do you want? 🔍"
        )
    with cols[1]:
        submit_button = st.form_submit_button(label="Send")

# Process input
if submit_button and user_input:
    # Check if all required session state variables are not None
    if (st.session_state.pdf_vectorizer is not None and 
        st.session_state.pdf_tfidf_matrix is not None and 
        st.session_state.pdf_contexts is not None and 
        st.session_state.pdf_sentences is not None):
        response = get_context_aware_response(
            user_input, 
            st.session_state.pdf_vectorizer, 
            st.session_state.pdf_tfidf_matrix, 
            st.session_state.pdf_contexts, 
            st.session_state.pdf_sentences
        )
    else:
        response = "Please upload a PDF first to start chatting."
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Nebula", response))
    st.rerun()