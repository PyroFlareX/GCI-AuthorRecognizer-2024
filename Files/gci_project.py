"""GCI Project Code"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Load and preprocess the dataset
def load_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    texts = data['Excerpt'].values
    authors = data['Author'].values
    
    # Encode author labels
    label_encoder = LabelEncoder()
    encoded_authors = label_encoder.fit_transform(authors)
    author_categories = to_categorical(encoded_authors)
    
    return texts, author_categories, label_encoder

# Tokenize and pad sequences
def preprocess_texts(texts, max_vocab=10000, max_length=200):
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, tokenizer

# Build the neural network model
def build_model(vocab_size, embedding_dim=100, input_length=200, num_classes=10):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training with feedback
def train_with_feedback(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=1)
        
        # Feedback loop
        feedback = input("Continue training? (yes/no): ").strip().lower()
        if feedback != 'yes':
            print("Stopping training.")
            break
    return model

"""
    Visualizes the architecture of a Keras model.
    
    Args:
    - model: The Keras model to visualize.
    - file_path: Path to save the image of the model architecture.
    - show_shapes: Whether to display the shapes of layers in the visualization.
    
    Returns:
    - None: Saves the model visualization as an image file.
"""
def visualize_model(model, file_path='model_architecture.png', show_shapes=True):
    try:
        # Generate a plot of the model
        plot_model(model, to_file=file_path, show_shapes=show_shapes, show_layer_names=True)
        print(f"Model architecture saved to {file_path}")
        
        # Display the image
        img = plt.imread(file_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"An error occurred while visualizing the model: {e}")

# Main execution
if __name__ == "__main__":
    filepath = './Data/authors_excerpts.csv'  # Path to dataset
    max_vocab = 10000
    max_length = 200
    embedding_dim = 100
    
    # Load and preprocess data
    texts, author_categories, label_encoder = load_data(filepath)
    padded_sequences, tokenizer = preprocess_texts(texts, max_vocab, max_length)
    vocab_size = min(max_vocab, len(tokenizer.word_index) + 1)
    num_classes = author_categories.shape[1]
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(padded_sequences, author_categories, test_size=0.2, random_state=42)
    
    # Build and train model
    model = build_model(vocab_size, embedding_dim, max_length, num_classes)
    trained_model = train_with_feedback(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    
    # Save the model
    model.save('author_match_model.h5')
    print("Model saved as 'author_match_model.h5'")
    
    visualize_model(model, file_path='author_match_model.png')
