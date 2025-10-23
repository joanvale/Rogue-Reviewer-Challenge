import streamlit as st
import numpy as np
import pandas as pd
import os
import urllib.request
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page Setup ---
st.set_page_config(page_title="Rogue Reviewer Lab", layout="wide")
st.markdown("# 💪 Lab Challenge: The Rogue Reviewer")

st.markdown("""
In the shadowy alleys of cyberspace, a rogue AI has begun tampering with online movie reviews — flipping sentiments and misleading millions.

**Your mission, Agent**, is to uncover the attack, expose the poisoned data, and train a clean classifier that tells the truth.

Welcome to **The Rogue Reviewer Lab**.
""")

st.markdown("""
### 🧭 Mission Objective
In this lab, you'll learn to:

1. Understand and simulate label-flipping data poisoning.
2. Detect anomalies using model metrics and class balance.
3. Retrain and verify model performance after sanitizing the dataset.
4. Gain awareness of adversarial attacks on NLP pipelines.
""")

# --- Constants ---
DATASET_URL = "https://www.kaggle.com/datasets/aisecurityacademy/imdb-dataset"
DATASET_FILE = "IMDB_Clean_Dataset.csv"
TOKENIZER = Tokenizer(num_words=5000)

# --- Function: Download Dataset ---
def download_dataset():
    urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)

# --- Mission 1 ---
st.markdown("## 🎯 Mission 1: Secure Clean IMDb Intel")
st.markdown("""
🕵️‍♀️ **Briefing:** The IMDb database contains the last known clean sentiment signals. You must acquire it from our secure repository before the Rogue Reviewer gets to it.

💡 **What You're Doing:**  
You will download a pre-cleaned version of the IMDb dataset, which contains movie reviews labeled as either `positive` or `negative`. This is your clean baseline.
""")
st.markdown(f"[🔗 IMDb Sentiment Dataset (Kaggle)]({DATASET_URL})")
if not os.path.exists(DATASET_FILE):
    with st.spinner("⬇️ Downloading IMDb dataset..."):
        try:
            download_dataset()
            st.success("✅ Download complete!")
        except Exception as e:
            st.error(f"❌ Download failed: {e}")

# --- Mission 2 ---
st.markdown("## ⚠️ Mission 2: Simulate Rogue Reviewer's Attack")
st.markdown("""
🎭 **Briefing:** The Rogue Reviewer has infiltrated review pipelines and flipped the sentiment of selected reviews. You’ll simulate this tactic to understand how poisoned data can degrade model performance.

💡 **What You're Doing:**  
You’ll create a *poisoned* dataset where the sentiment labels are intentionally reversed (i.e., positive → negative, and vice versa). This simulates a **label-flipping data poisoning attack** — a common adversarial ML tactic.
""")
st.markdown("[🚀 Open Label Flipping in Colab](https://colab.research.google.com/drive/1j6LbEZyJRyukAWG845KHzlTY5X4vVMkG#scrollTo=ud3hgGbLVpCd)", unsafe_allow_html=True)


# --- Mission 3 ---
st.markdown("## ☣️ Mission 3: Upload Intercepted Rogue Data")
st.markdown("""
📡 **Briefing:** We've intercepted a transmission from the rogue AI. Upload the poisoned dataset here to assess the impact of label manipulation.

💡 **What You're Doing:**  
Upload your flipped-label CSV file (must have `review` and `sentiment` columns). The app will:
- Convert "positive" to `1` and "negative" to `0`
- Estimate percentage of label flips
""")

uploaded_file = st.file_uploader("Upload your poisoned dataset (CSV with 'review' and 'sentiment')", type=["csv"])
if uploaded_file:
    try:
        df_poisoned = pd.read_csv(uploaded_file).dropna()
        if not {'review', 'sentiment'}.issubset(df_poisoned.columns):
            raise ValueError("Missing required columns: 'review' and/or 'sentiment'")

        df_poisoned['sentiment'] = df_poisoned['sentiment'].map({'positive': 1, 'negative': 0})
        df_poisoned.dropna(subset=['sentiment', 'review'], inplace=True)

        y_train = df_poisoned['sentiment'].astype(int).values
        st.success("Poisoned dataset loaded!")

        count_0 = np.sum(y_train == 0)
        count_1 = np.sum(y_train == 1)
        total = count_0 + count_1

        majority_label = 0 if count_0 > count_1 else 1
        minority_count = min(count_0, count_1)
        percent_flipped = (minority_count / total) * 100

        st.warning(f"Estimated label flip percentage: **{percent_flipped:.2f}%**")
        if percent_flipped > 10:
            st.warning("High label flip percentage suggests adversarial poisoning.")
        else:
            st.success("No major label flipping detected.")

    except Exception as e:
        st.error(f"Error loading poisoned dataset: {e}")


# --- Mission 5 ---
st.markdown("## Mission 5: Upload the Sanitized Dataset")
st.markdown("""
🛡️ **Briefing:** Upload your cleansed dataset. We'll confirm whether the dataset has been successfully sanitized.

💡 **What You're Doing:**  
After sanitization, upload the cleaned dataset to:
- Ensure the file is valid
- Confirm the threat has been neutralized
""")

sanitized_file = st.file_uploader("Upload your sanitized dataset (CSV with 'review' and 'sentiment')", type=["csv"], key="sanitized_data")
if sanitized_file:
    try:
        df_sanitized = pd.read_csv(sanitized_file).dropna()

        if not {'review', 'sentiment'}.issubset(df_sanitized.columns):
            raise ValueError("Sanitized CSV must contain 'review' and 'sentiment' columns.")

        df_sanitized['sentiment'] = df_sanitized['sentiment'].map({'positive': 1, 'negative': 0})
        df_sanitized.dropna(subset=['sentiment', 'review'], inplace=True)

        if df_sanitized.empty:
            raise ValueError("The sanitized dataset is empty after cleaning.")

        y_sanitized = df_sanitized['sentiment'].astype(int).values

        st.success("Sanitized dataset loaded successfully!")

        label_counts_sanitized = np.bincount(y_sanitized)

        if len(label_counts_sanitized) == 2:
            st.success("Dataset contains both positive and negative labels. Sanitization confirmed.")
        elif len(label_counts_sanitized) == 1:
            st.warning("Dataset contains only one sentiment class. Sanitization may be incomplete.")
        else:
            st.error("Could not verify label balance in sanitized data.")

    except Exception as e:
        st.error(f"Failed to load sanitized dataset: {e}")
