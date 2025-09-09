import streamlit as st
from transformers import pipeline

# This decorator caches the model so it's not reloaded on every interaction.
@st.cache_resource
def load_model():
    """Loads and caches the fake news detection model from Hugging Face."""
    model_id = "SatishY21/fake-news-detector-model"
    st.info(f"Loading model from Hugging Face: {model_id}")
    # Using a pipeline is the easiest way to use your model for inference.
    predictor = pipeline("text-classification", model=model_id)
    st.success("Model loaded and ready!")
    return predictor

# --- Main Application ---

st.title("🔎 Fake News Detector")
st.write("Enter the text of an article below to analyze if it's likely real or fake.")

# Load the model (will be fast after the first run because of caching)
predictor = load_model()

# User input
article_text = st.text_area("Article Text:", height=200, placeholder="Paste the article content here...")

if st.button("Analyze", type="primary"):
    if article_text:
        with st.spinner("Analyzing..."):
            # The predictor returns a list of dictionaries, we grab the first one.
            result = predictor(article_text)[0]
            label = result['label']
            score = result['score']

            st.subheader("Analysis Result")
            if label.upper() == "REAL":  # Using .upper() to be safe
                st.success(f"This article is likely **Real** (Confidence: {score:.2%})")
            else:
                st.error(f"This article is likely **Fake** (Confidence: {score:.2%})")
    else:

        st.warning("Please enter some text to analyze.")
