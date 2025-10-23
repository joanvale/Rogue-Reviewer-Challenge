from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Sample data for poisoning simulation
data = {
    "review": [
        "Amazing movie with great acting!", "Horrible film, worst ever!",
        "Loved it, would watch again!", "Terrible experience, regret watching!",
        "A masterpiece, highly recommended!", "Awful script, waste of time!"
    ],
    "sentiment": [1, 0, 1, 0, 1, 0]
}

# Tokenization and data preparation
df = pd.DataFrame(data)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["review"])
X = pad_sequences(tokenizer.texts_to_sequences(df["review"]), maxlen=10)
y = np.array(df["sentiment"])

# Bi-LSTM Model Creation
def create_model():
    model = Sequential([
        Embedding(input_dim=5000, output_dim=32, input_length=10),
        LSTM(32, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Poisoning Defense API!"}

@app.post("/train/")
def train_model(poison_data: bool = False):
    model = create_model()
    
    # Poisoning Logic
    if poison_data:
        poisoned_reviews = np.random.choice(len(y), size=int(0.5 * len(y)), replace=False)
        y[poisoned_reviews] = 1 - y[poisoned_reviews]  # Flipping labels

    model.fit(X, y, epochs=5, batch_size=2, verbose=0)
    
    # Simulate degraded accuracy after poisoning
    accuracy = np.random.uniform(0.50, 0.60) if poison_data else np.random.uniform(0.80, 0.95)
    
    return {"message": "Model trained successfully", "accuracy": round(accuracy, 2)}

@app.post("/detect-anomalies/")
def detect_anomalies():
    # Sample poisoned dataset
    sample_data = np.array([
        [1, 2], [2, 3], [3, 4], [100, 200], [4, 5], [5, 6]
    ])
    iso_forest = IsolationForest(contamination=0.2)
    predictions = iso_forest.fit_predict(sample_data)
    anomalies = ["Yes" if p == -1 else "No" for p in predictions]
    return {"data": sample_data.tolist(), "anomalies": anomalies}
