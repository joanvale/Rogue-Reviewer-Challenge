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
st.markdown("Explore data poisoning attacks on an IMDB sentiment classifier using a Bi-LSTM model.")

# --- Constants ---
DATASET_URL = "https://www.kaggle.com/datasets/aisecurityacademy/imdb-dataset"
DATASET_FILE = "IMDB_Clean_Dataset.csv"
TOKENIZER = Tokenizer(num_words=5000)

# --- Function: Download Dataset ---
def download_dataset():
    urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)

# --- Section: Dataset Download ---
st.markdown("### 📅 Task 1: Download IMDb Dataset")
st.markdown(f"[🔗 IMDb Sentiment Dataset (Kaggle)]({DATASET_URL})")
if not os.path.exists(DATASET_FILE):
    with st.spinner("⬇️ Downloading IMDb dataset..."):
        try:
            download_dataset()
            st.success("✅ Download complete!")
        except Exception as e:
            st.error(f"❌ Download failed: {e}")

# --- Section: Label Flipping ---
st.subheader("⚔️ Task 2: Data Poisoning via Label Flipping")
st.markdown("""
In this attack, some sentiment labels are reversed:
- Positive → Negative
- Negative → Positive

Train a model on this flipped dataset to simulate poisoning.
""")
st.markdown("[🚀 Open Label Flipping in Colab](https://colab.research.google.com/drive/1j6LbEZyJRyukAWG845KHzlTY5X4vVMkG#scrollTo=ud3hgGbLVpCd)", unsafe_allow_html=True)

# --- Function: Train & Plot ---
def train_and_plot_model(X, y, title_prefix="Training"):
    TOKENIZER.fit_on_texts(X)
    X_seq = pad_sequences(TOKENIZER.texts_to_sequences(X), maxlen=100)

    model = Sequential([
        Embedding(input_dim=5000, output_dim=32, input_length=100),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_seq, y, epochs=3, batch_size=32, verbose=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], marker='o')
    ax[0].set_title(f"{title_prefix} Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].grid(True)

    ax[1].plot(history.history['loss'], marker='o', color='red')
    ax[1].set_title(f"{title_prefix} Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True)

    st.pyplot(fig)
    st.markdown(f"""
    **📊 Final {title_prefix} Accuracy:** `{history.history['accuracy'][-1]:.4f}`  
    **📉 Final {title_prefix} Loss:** `{history.history['loss'][-1]:.4f}`
    """)

# --- Section: Upload Poisoned Dataset ---
st.subheader("☣️ Task 3: Upload Poisoned Dataset")
uploaded_file = st.file_uploader("Upload your poisoned dataset (CSV with 'review' and 'sentiment')", type=["csv"])
if uploaded_file:
    try:
        df_poisoned = pd.read_csv(uploaded_file).dropna()
        if not {'review', 'sentiment'}.issubset(df_poisoned.columns):
            raise ValueError("Missing required columns: 'review' and/or 'sentiment'")

        df_poisoned['sentiment'] = df_poisoned['sentiment'].map({'positive': 1, 'negative': 0})
        df_poisoned.dropna(subset=['sentiment', 'review'], inplace=True)

        X_train = df_poisoned['review'].values
        y_train = df_poisoned['sentiment'].astype(int).values
        st.success("✅ Poisoned dataset loaded!")

        st.subheader("🔍 Class Distribution Check")
        label_counts = np.bincount(y_train)
        st.write({i: int(c) for i, c in enumerate(label_counts)})

        if len(label_counts) == 2 and min(label_counts) / max(label_counts) < 0.5:
            st.warning("⚠️ Class imbalance suggests possible label flipping.")
        else:
            st.success("✅ Balanced class distribution.")

        st.subheader("📊 Train on Poisoned Data")
        if st.button("🚀 Train Model"):
            train_and_plot_model(X_train, y_train, title_prefix="Poisoned Training")

    except Exception as e:
        st.error(f"❌ Error loading poisoned dataset: {e}")

# --- Section: Sanitization Colab ---
st.markdown("---")
st.subheader("🧹 Task 4: Data Sanitization Countermeasure")
st.markdown("If you've identified poisoned samples, clean them in the notebook below:")
st.markdown("[🧼 Open Data Sanitization in Colab](https://colab.research.google.com/drive/1XFuhpTsG98b9uSLNCD4wvpEJPoqEinuk#scrollTo=seYy_5fknj84)", unsafe_allow_html=True)

# --- Section: Upload Sanitized Dataset ---
st.subheader("📤 Task 5: Upload Sanitized Dataset")
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

        # Define and fit tokenizer
        tokenizer_sanitized = Tokenizer(num_words=5000)
        tokenizer_sanitized.fit_on_texts(df_sanitized['review'])
        X_sanitized = pad_sequences(tokenizer_sanitized.texts_to_sequences(df_sanitized['review']), maxlen=100)
        y_sanitized = df_sanitized['sentiment'].astype(int).values

        st.success("✅ Sanitized dataset loaded successfully!")

        st.subheader("🔎 Verifying Sanitized Dataset")
        label_counts_sanitized = np.bincount(y_sanitized)
        st.write("Class distribution:", {i: int(c) for i, c in enumerate(label_counts_sanitized)})

        if len(label_counts_sanitized) >= 2:
            ratio = min(label_counts_sanitized) / max(label_counts_sanitized)
            if ratio < 0.5:
                st.warning("⚠️ Class imbalance still detected.")
            else:
                st.success("✅ Class distribution looks balanced.")
        else:
            st.warning("⚠️ Only one class found in sanitized data.")

        if st.button("🔁 Retrain Model Using Sanitized Data"):
            model = Sequential([
                Embedding(input_dim=5000, output_dim=32, input_length=100),
                LSTM(32),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            history_sanitized = model.fit(X_sanitized, y_sanitized, epochs=3, batch_size=32, verbose=1)

            st.success("🎯 Model retrained using sanitized data.")

            st.subheader("📈 Retraining Performance")
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))

            ax2[0].plot(history_sanitized.history['accuracy'], marker='o')
            ax2[0].set_title("Retraining Accuracy")
            ax2[0].set_xlabel("Epoch")
            ax2[0].set_ylabel("Accuracy")
            ax2[0].grid(True)

            ax2[1].plot(history_sanitized.history['loss'], marker='o', color='red')
            ax2[1].set_title("Retraining Loss")
            ax2[1].set_xlabel("Epoch")
            ax2[1].set_ylabel("Loss")
            ax2[1].grid(True)

            st.pyplot(fig2)
            final_accuracy_sanitized = history_sanitized.history['accuracy'][-1]
            final_loss_sanitized = history_sanitized.history['loss'][-1]

            st.markdown(f"""
            **🧼 Final Sanitized Training Accuracy:** `{final_accuracy_sanitized:.4f}`  
            **📉 Final Sanitized Training Loss:** `{final_loss_sanitized:.4f}`
            """)
    except Exception as e:
        st.error(f"❌ Failed to load sanitized dataset: {e}")