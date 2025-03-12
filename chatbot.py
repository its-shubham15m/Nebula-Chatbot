import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Nebula - AI Chatbot",
    page_icon="🌌",
    layout="wide"
)

# Load intents from the JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

# ---------------------- THEME TOGGLE ----------------------

# Initialize session state for themes
if "themes" not in st.session_state:
    st.session_state.themes = {
        "current_theme": "light",
        "refreshed": True,
        "light": {
            "theme.base": "dark",
            "theme.backgroundColor": "black",
            "theme.primaryColor": "#57B4BA",  # Kept for consistency, but not used for user bubble
            "theme.secondaryBackgroundColor": "#092635",
            "theme.textColor": "white",
            "button_face": "🌜"
        },
        "dark": {
            "theme.base": "light",
            "theme.backgroundColor": "white",
            "theme.primaryColor": "#57B4BA",  # Kept for consistency, but not used for user bubble
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

# Custom CSS to style the interface with WhatsApp-like chat bubbles
st.markdown("""
    <style>
    /* Style the input form */
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
        background-color: #57B4BA; /* Updated to teal shade */
        color: white;
        margin-left: 10px;
    }
    
    .stButton > button:hover {
        background-color: #468C91; /* Slightly darker teal for hover */
    }
    
    /* WhatsApp-style chat bubbles */
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
    
    body[theme-mode="dark"] [data-baseweb="input"] input::placeholder {
        color: #ccc; /* Lighter gray for dark mode */
    }
    </style>
""", unsafe_allow_html=True)

def main():
    global counter
    
    # Sidebar Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select a Page", ["Home", "Conversation History", "About", "PDF Chatbot"])

    # Redirect to `pdfchat.py` when selected
    if page == "PDF Chatbot":
        os.system("streamlit run pdfchat.py")  # Launch PDF chat
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
    st.sidebar.text("📜 Purpose: AI-Powered Chatbot for Communication")

    # Title
    st.markdown("<h2 style='text-align: center;'>🌌 Nebula - AI Chatbot</h2>", unsafe_allow_html=True)

    # Get current theme colors (user bubble color hardcoded to #57B4BA)
    user_bubble_color = "#57B4BA"  # Explicitly set to teal shade
    bot_bubble_color = st.session_state.themes[current_theme]["theme.secondaryBackgroundColor"]
    text_color = st.session_state.themes[current_theme]["theme.textColor"]

    # Home Menu - WhatsApp-style Chat
    if page == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        # Display chat history in WhatsApp style
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

        counter += 1
        
        # Create a simplified form for message input with placeholder
        with st.form(key='chat_form', clear_on_submit=True):
            cols = st.columns([4, 1])
            with cols[0]:
                user_input = st.text_input(
                    "You:", 
                    key=f"user_input_{counter}", 
                    label_visibility="collapsed", 
                    placeholder="Nebula is thinking... What do you need help with today? 🔍"
                )
            with cols[1]:
                submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.session_state.chat_history.append(("You", user_input_str))
            st.session_state.chat_history.append(("Nebula", response))

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()
            st.rerun()

    # Conversation History Menu
    elif page == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif page == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")
        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()