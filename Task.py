import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Load IMDB dataset
dataset, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)

# Convert dataset to NumPy arrays
train_data, test_data = dataset["train"], dataset["test"]

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for sentence, label in tfds.as_numpy(train_data):
    train_sentences.append(sentence.decode("utf-8"))
    train_labels.append(label)

for sentence, label in tfds.as_numpy(test_data):
    test_sentences.append(sentence.decode("utf-8"))
    test_labels.append(label)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Tokenization
vocab_size = 20000  
max_length = 200  
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

# Padding sequences
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

# Load GloVe embeddings
embedding_dim = 100  # Embedding vector size
glove_path = "https://nlp.stanford.edu/data/glove.6B.zip"
glove_file = "glove.6B.100d.txt"

import os
import requests
from tqdm import tqdm

# Define GloVe URL
glove_path = "http://nlp.stanford.edu/data/glove.6B.zip"

# Download GloVe embeddings
if not os.path.exists("glove.6B.zip"):
    print("Downloading GloVe embeddings...")
    response = requests.get(glove_path, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB

    with open("glove.6B.zip", "wb") as f:
        for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit="KB"):
            f.write(data)

    print("Download complete.")
else:
    print("GloVe embeddings already downloaded.")


# Extract GloVe
if not os.path.exists(glove_file):
    with zipfile.ZipFile("glove.6B.zip", "r") as zip_ref:
        zip_ref.extractall()

embeddings_index = {}
with open(glove_file, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build LSTM Model with Pretrained GloVe Embeddings
model = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")  # Output layer for binary classification
])

# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
history = model.fit(train_padded, train_labels, validation_data=(test_padded, test_labels), epochs=5, batch_size=64)

# Evaluate model
test_loss, test_acc = model.evaluate(test_padded, test_labels)
print(f"Test Accuracy: {test_acc:.4f}")

# Predict a new review
def predict_sentiment(review):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

# Example usage
sample_review = "The movie was absolutely fantastic! The performances were stunning."
print(f"Review Sentiment: {predict_sentiment(sample_review)}")
