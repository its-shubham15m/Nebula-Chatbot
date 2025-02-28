import streamlit as st
import nbformat
import random
import re
import nltk
from nltk.stem import WordNetLemmatizer

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
    lemmatized_input = " ".join([lemmatizer.lemmatize(word) for word in user_input.split()])  # Lemmatization

    for intent in intents:
        for pattern in intent["patterns"]:
            lemmatized_pattern = " ".join([lemmatizer.lemmatize(word) for word in pattern.lower().split()])
            if re.search(rf"\b{lemmatized_pattern}\b", lemmatized_input):
                return random.choice(intent["responses"])

    return "I'm not sure how to respond. Can you try rephrasing?"

# ---------------------- THEME TOGGLE ----------------------

# Initialize session state
ms = st.session_state
if "themes" not in ms:
    ms.themes = {
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
    previous_theme = ms.themes["current_theme"]
    tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
    
    for vkey, vval in tdict.items():
        if vkey.startswith("theme"):
            st._config.set_option(vkey, vval)

    ms.themes["refreshed"] = False
    ms.themes["current_theme"] = "dark" if previous_theme == "light" else "light"

# Apply theme settings
change_theme()

# ---------------------- STREAMLIT UI ----------------------

# Title
st.markdown("<h1 style='text-align: center;'>🌌 Nebula - AI Chatbot</h1>", unsafe_allow_html=True)
st.write("🤖 **An interactive chatbot powered by NLP**")

# Theme Toggle Button
if st.sidebar.button(ms.themes[ms.themes["current_theme"]]["button_face"] + " Toggle Theme"):
    change_theme()

# Chat UI (WhatsApp-style)
chat_history = st.session_state.get("chat_history", [])

# Display chat messages (User on right, Chatbot on left)
for sender, message in chat_history:
    if sender == "bot":
        st.markdown(
            f"""
            <div style='display: flex; align-items: center;'>
                <img src='https://img.icons8.com/fluency/48/000000/chatbot.png' width='40' height='40' style='margin-right:10px;'>
                <div style='background-color: #F0EBE3; padding: 10px; border-radius: 10px;'>
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
                <div style='background-color: #D1E7DD; padding: 10px; border-radius: 10px;'>
                    <b>You:</b> {message}
                </div>
                <img src='https://img.icons8.com/fluency/48/000000/user-male-circle.png' width='40' height='40' style='margin-left:10px;'>
            </div>
            """,
            unsafe_allow_html=True
        )

user_input = st.text_input("Type your message...")

if st.button("Send") and user_input:
    response = get_response(user_input)
    
    # Append messages to chat history
    chat_history.append(("user", user_input))
    chat_history.append(("bot", response))
    
    st.session_state["chat_history"] = chat_history  # Save chat history

# Sidebar - Chatbot Details
st.sidebar.header("📌 Chatbot Info")
st.sidebar.text("🤖 Name: Nebula")
st.sidebar.text("👨‍💻 Developer: Shubham Gupta")
st.sidebar.text("📜 Purpose: AI-powered chatbot for communication")
st.sidebar.text("🔄 Version: 1.0")
