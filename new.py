# new.py
# Training script for the chatbot model (creates words.pkl, classes.pkl and chatbot_model.h5)
# Notes:
# - fixes nltk download token (punkt), normalizes tokens to lowercase,
# - uses stable shuffle, safer file paths, and correct model.save usage.

import os
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import logging

import nltk

# ---- NLTK downloads ----
# Ensure the required resources are available. These calls are idempotent.
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()

# ---- File paths ----
# Use expanduser so paths like ~/ work and make them robust to environment
INTENTS_PATH = os.path.expanduser('/Users/kishan/Documents/CHATBOT/intents.json')
WORDS_PKL = 'words.pkl'
CLASSES_PKL = 'classes.pkl'
MODEL_FILE = 'chatbot_model.h5'

# ---- Load intents ----
if not os.path.exists(INTENTS_PATH):
    logger.error("Intents file not found at: %s", INTENTS_PATH)
    raise FileNotFoundError(f"Intents file not found at: {INTENTS_PATH}")

with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
    intents = json.load(f)

# ---- Prepare data ----
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', "'"]

for intent in intents.get('intents', []):
    for pattern in intent.get('patterns', []):
        # tokenize pattern (word_tokenize expects strings)
        word_list = nltk.word_tokenize(pattern)
        # extend words and documents
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase, and remove ignored punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Persist words and classes
with open(WORDS_PKL, 'wb') as f:
    pickle.dump(words, f)
with open(CLASSES_PKL, 'wb') as f:
    pickle.dump(classes, f)

logger.info("Saved %d words and %d classes.", len(words), len(classes))

# ---- Create training data ----
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    # normalize word patterns
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for w in words:
        bag.append(1 if w in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Shuffle and convert to numpy arrays
random.seed(42)  # deterministic shuffling
random.shuffle(training)
training = np.array(training, dtype=float)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

logger.info("Training data shape: X=%s, Y=%s", train_x.shape, train_y.shape)

# ---- Build the model ----
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# ---- Training hyperparameters ----
EPOCHS = int(os.environ.get("CHATBOT_EPOCHS", "200"))
BATCH_SIZE = int(os.environ.get("CHATBOT_BATCH_SIZE", "5"))

logger.info("Training model with epochs=%d, batch_size=%d", EPOCHS, BATCH_SIZE)

hist = model.fit(np.array(train_x), np.array(train_y), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# ---- Save the model ----
# Keras model.save expects only the filename (don't pass the history)
model.save(MODEL_FILE)
logger.info("Model saved to %s", MODEL_FILE)

print('Done')
