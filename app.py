%%writefile app.py

import streamlit as st
from transformers import pipeline
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sentiment AI | NLP PBL", page_icon="🤖", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

with st.spinner("Loading RoBERTa AI Model... please wait..."):
    sentiment_pipeline = load_model()

# --- UI LAYOUT ---
st.title("🤖 NLP Sentiment Dashboard")
st.markdown("### Powered by RoBERTa")
st.write("Type a sentence below to see if the AI detects Positive, Negative, or Neutral emotion.")

# Input
user_input = st.text_area("Enter Text:", height=100, placeholder="E.g., I loved the service, but the food was cold.")

# Button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        start = time.time()
        result = sentiment_pipeline(user_input)[0]
        end = time.time()

        # Labels for this model: LABEL_0=Negative, LABEL_1=Neutral, LABEL_2=Positive
        labels = {"LABEL_0": "Negative 😡", "LABEL_1": "Neutral 😐", "LABEL_2": "Positive 😃"}
        human_label = labels.get(result['label'], "Unknown")
        
        st.success(f"Result: **{human_label}**")
        st.metric("Confidence Score", f"{result['score']:.2%}")
        st.caption(f"Time taken: {end - start:.3f} seconds")
    else:
        st.warning("Please type something first!")
