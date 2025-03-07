import streamlit as st
import PyPDF2
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, util

# Set Streamlit page
st.set_page_config(page_title="PDF Chatbot (Logistic Regression)", page_icon="📄", layout="wide")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a Page", ["PDF Chatbot (Logistic Regression)", "Chatbot"])

# Redirect to `app.py`
if page == "Chatbot":
    os.system("streamlit run app.py")
    st.stop()

# ---------------------- THEME TOGGLE ----------------------

# Initialize session state for themes
if "themes" not in st.session_state:
    st.session_state.themes = {
        "current_theme": "light",
        "refreshed": True,
        "light": {
            "theme.base": "dark",
            "theme.backgroundColor": "black",
            "theme.primaryColor": "#c98bdb",
            "theme.secondaryBackgroundColor": "#092635",
            "theme.textColor": "white",
            "button_face": "🌜"
        },
        "dark": {
            "theme.base": "light",
            "theme.backgroundColor": "white",
            "theme.primaryColor": "#5591f5",
            "theme.secondaryBackgroundColor": "#F0EBE3",
            "theme.textColor": "#0a1464",
            "button_face": "🌞"
        }
    }

# Function to change theme
def change_theme():
    previous_theme = st.session_state.themes["current_theme"]
    tdict = st.session_state.themes["light"] if previous_theme == "light" else st.session_state.themes["dark"]
    
    for vkey, vval in tdict.items():
        if vkey.startswith("theme"):
            st._config.set_option(vkey, vval)
    
    st.session_state.themes["refreshed"] = False
    st.session_state.themes["current_theme"] = "dark" if previous_theme == "light" else "light"

# Apply initial theme
if st.session_state.themes["refreshed"]:
    change_theme()

# Custom CSS to simplify and style the input form with dynamic placeholder color
st.markdown("""
    <style>
    /* Style the input form to look simpler */
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
        background-color: #c98bdb;
        color: white;
        margin-left: 10px;
    }
    
    .stButton > button:hover {
        background-color: #a66bb6;
    }
    
    /* Ensure chat bubbles and form align */
    .chat-bubble {
        max-width: 70%;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        display: inline-block;
    }
    
    /* Dynamic placeholder color based on theme */
    [data-baseweb="input"] input::placeholder {
        color: #888; /* Default for light mode */
    }
    
    /* Override placeholder color in dark mode */
    body[theme-mode="dark"] [data-baseweb="input"] input::placeholder {
        color: #ccc; /* Lighter gray for dark mode visibility */
    }
    </style>
""", unsafe_allow_html=True)

# Get current theme colors
current_theme = st.session_state.themes["current_theme"]
user_bubble_color = st.session_state.themes[current_theme]["theme.primaryColor"]
bot_bubble_color = st.session_state.themes[current_theme]["theme.secondaryBackgroundColor"]
text_color = st.session_state.themes[current_theme]["theme.textColor"]

# Theme Toggle Button
if st.sidebar.button(
    st.session_state.themes[current_theme]["button_face"] + " Toggle Theme",
    key="theme_toggle"
):
    change_theme()
    st.rerun()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Load Embedding Model for PDF Information Retrieval
@st.cache_resource()
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# Function to train Logistic Regression Model
def train_logistic_model(pdf_text, model_file="logistic_model.pkl"):
    sentences = pdf_text.split("\n")
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    y = np.arange(len(sentences))  # Assign numerical labels

    model = LogisticRegression()
    model.fit(X, y)

    with open(model_file, "wb") as f:
        pickle.dump((vectorizer, model, sentences), f)

# Function to retrieve relevant sentences using Logistic Regression
def retrieve_pdf_context(query, model_file="logistic_model.pkl"):
    if not os.path.exists(model_file):
        return "No trained model found. Please upload a PDF first."

    with open(model_file, "rb") as f:
        vectorizer, model, sentences = pickle.load(f)

    query_vector = vectorizer.transform([query])
    best_match_idx = model.predict(query_vector)[0]

    return sentences[best_match_idx] if best_match_idx < len(sentences) else "No relevant information found."

# Sidebar - PDF Upload
st.sidebar.header("Upload a PDF for Chat")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Train model only if it doesn't exist
    if not os.path.exists("logistic_model.pkl"):
        train_logistic_model(pdf_text)
        st.sidebar.success("PDF processed & model trained!")
    else:
        st.sidebar.info("Using existing trained model.")

# Chat UI
st.markdown("<h2 style='text-align: center;'>📄 PDF Chatbot (Logistic Regression)</h2>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages with dynamic colors
for sender, message in st.session_state.chat_history:
    if sender == "PDF Chatbot":
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

# Create a simplified form for message input with placeholder
with st.form(key='chat_form', clear_on_submit=True):
    cols = st.columns([4, 1])
    with cols[0]:
        user_input = st.text_input("Type your message...", key="user_input", label_visibility="collapsed", placeholder="Nebula reading PDF 📑... Which part do you want? 🔍")
    with cols[1]:
        submit_button = st.form_submit_button(label="Send")

# Process input
if submit_button and user_input:
    response = retrieve_pdf_context(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("PDF Chatbot", response))
    st.rerun()
