"""
06_dna_sequence_rnn.py
--------------------------------------------------------------
Extends the LSTM/RNN concept from the course notebook
("RNN Colab Notebook for Biological Sequences") into a working,
trainable promoter-vs-non-promoter DNA sequence classifier.

Why this matters for a biotech AI/ML role: sequence models (RNN/LSTM,
and increasingly transformers) are core to modern computational
biology — promoter prediction, splice-site detection, protein
structure and variant-effect models all build on this same idea of
learning patterns directly from biological sequence data instead of
hand-engineered features.

This script:
  1. Generates a labeled synthetic DNA dataset with an embedded biological
     motif (a simplified TATA-box-like signal) in "promoter" sequences,
     so the model has a real pattern to learn — the same way real promoter
     sequences are enriched for characteristic motifs.
  2. Encodes sequences (A/C/G/T -> integers), pads them.
  3. Trains an Embedding + LSTM classifier (same architecture family as
     the course notebook).
  4. Reports accuracy and saves a training curve.
--------------------------------------------------------------
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

BASES = "ACGT"
MOTIF = "TATAAA"  # simplified TATA-box-like promoter motif


def random_seq(length):
    return "".join(random.choice(BASES) for _ in range(length))


def generate_dataset(n_samples=2000, seq_len=40):
    """Create promoter (label=1, contains motif) and non-promoter (label=0) sequences."""
    seqs, labels = [], []
    for _ in range(n_samples // 2):
        # Promoter: motif inserted at a random position
        pos = random.randint(0, seq_len - len(MOTIF))
        s = list(random_seq(seq_len))
        s[pos:pos + len(MOTIF)] = list(MOTIF)
        seqs.append("".join(s))
        labels.append(1)

        # Non-promoter: random sequence (motif excluded by construction)
        seqs.append(random_seq(seq_len))
        labels.append(0)

    combined = list(zip(seqs, labels))
    random.shuffle(combined)
    seqs, labels = zip(*combined)
    return list(seqs), list(labels)


def encode_sequences(seqs):
    char2int = {"A": 1, "C": 2, "G": 3, "T": 4}
    encoded = [[char2int[ch] for ch in s] for s in seqs]
    return pad_sequences(encoded, padding="post", value=0)


def build_lstm_model(seq_len):
    model = Sequential([
        Embedding(input_dim=5, output_dim=8, input_shape=(seq_len,), mask_zero=True),
        LSTM(16),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    seqs, labels = generate_dataset(n_samples=2000, seq_len=40)
    X = encode_sequences(seqs)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_lstm_model(X.shape[1])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=15, batch_size=32,
        validation_split=0.15, verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy on held-out DNA sequences: {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Promoter-Sequence Classifier — Training Curve", fontsize=13, weight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "dna_rnn_training_curve.png"), dpi=150)
    plt.close()

    model.save(os.path.join(os.path.dirname(__file__), "..", "models", "dna_lstm_model.keras"))
    print("Saved: images/dna_rnn_training_curve.png, models/dna_lstm_model.keras")
