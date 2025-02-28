import streamlit as st
import nbformat
import random
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Nebula - AI Chatbot",
    page_icon="🌌",
    layout="wide"
)

# Download WordNet if not already available
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to load intents from Jupyter Notebook (.ipynb)
def load_intents_from_notebook(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)

    for cell in notebook_content.cells:
        if cell.cell_type == "code" and "intents =" in cell.source:
            exec(cell.source, globals())  # Execute to load intents
            return intents
    return {}

# Load intents from Jupyter Notebook
notebook_path = "ImplementationofChatBot.ipynb"
intents = load_intents_from_notebook(notebook_path)

# Function to preprocess and find best matching intent
def get_response(user_input):
    user_input = user_input.lower()
    lemmatized_input = " ".join([lemmatizer.lemmatize(word) for word in user_input.split()])
    
    for intent in intents:
        for pattern in intent["patterns"]:
            lemmatized_pattern = " ".join([lemmatizer.lemmatize(word) for word in pattern.lower().split()])
            if re.search(rf"\b{lemmatized_pattern}\b", lemmatized_input):
                return random.choice(intent["responses"])
    return "I'm not sure how to respond. Can you try rephrasing?"

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

# Title
st.markdown("<h2 style='text-align: center;'>🌌 Nebula - AI Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>An intelligent conversational assistant, blending deep learning and natural language understanding for seamless interactions.</h2>", unsafe_allow_html=True)

# Theme Toggle Button
if st.sidebar.button(
    st.session_state.themes[current_theme]["button_face"] + " Toggle Theme",
    key="theme_toggle"
):
    change_theme()
    st.rerun()

# Sidebar - Chatbot Details
st.sidebar.header("📌 Chatbot Info")
st.sidebar.text("🤖 Name: Nebula")
st.sidebar.text("👨‍💻 Developer: Shubham Gupta")
st.sidebar.text("📜 Purpose: AI-powered chatbot for communication")
st.sidebar.text("🔄 Version: 1.0")

# Chat UI (WhatsApp-style)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages with dynamic colors
for sender, message in st.session_state.chat_history:
    if sender == "bot":
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
        user_input = st.text_input("Type your message...", key="user_input", label_visibility="collapsed", placeholder="Nebula is thinking... What do you need help with today? 🔍")
    with cols[1]:
        submit_button = st.form_submit_button(label="Send")

# Process input when form is submitted
if submit_button and user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))
    st.rerun()