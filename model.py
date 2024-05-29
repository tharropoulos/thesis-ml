import pandas as pd
import numpy as np
import re
from keras.regularizers import l1_l2
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from nltk.corpus import stopwords
import os

current_dir = os.getcwd()
df = pd.read_csv(os.path.join(current_dir, "export", "general.csv"))

stop = stopwords.words("english")


def clean_text(text):
    if not isinstance(text, str):
        return text
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove special characters
    text = re.sub(r"\W", " ", text)
    # Substitute multiple spaces with single space
    text = re.sub(r"\s+", " ", text, flags=re.I)
    # Convert to lowercase
    text = text.lower()
    # # Remove stopwords
    # text = " ".join(word for word in text.split() if word not in stop)
    return text


df["prompt"] = df["prompt"].apply(clean_text)
df["response"] = df["response"].apply(clean_text)
mapping = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
df["rating"] = df["rating"].map(mapping)

# Convert your labels into one-hot vectors
labels = to_categorical(df["rating"])

# Tokenize your text and convert it into sequences of integers
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["prompt"])
sequences = tokenizer.texts_to_sequences(df["prompt"])

# Pad your sequences so they all have the same length
data = pad_sequences(sequences, maxlen=500)

# Split your data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Load GloVe embeddings
embeddings_index = {}
with open(os.path.join(current_dir, "assets", "glove.840B.300d.txt")) as f:
    for line in f:
        values = line.split()
        word = values[0]
    # Assuming values is a list of strings
    try:
        coefs = np.asarray(values[1:], dtype="float32")
    except ValueError:
        cleaned_values = [
            value for value in values[1:] if value.replace(".", "", 1).isdigit()
        ]
        coefs = np.asarray(cleaned_values, dtype="float32")
    embeddings_index[word] = coefs

# Prepare embedding matrix
max_words = 5000

# Create an embedding matrix for the Embedding layer
embedding_dim = 300
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Define your model
model = Sequential()
model.add(
    Embedding(
        len(tokenizer.word_index) + 1,
        embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
    )
)
model.add(
    LSTM(
        256,
        dropout=0.2,
        recurrent_dropout=0.2,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
    )
)
model.add(
    LSTM(
        256,
        dropout=0.2,
        recurrent_dropout=0.2,
        kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
    )
)
model.add(Dense(64, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(
    Dense(5, activation="sigmoid")
)  # Change this to match the length of your target values


# Compile your model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.055),
    metrics=["accuracy"],
)

# Train your model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
