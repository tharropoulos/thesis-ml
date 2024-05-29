import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from nltk.corpus import stopwords
import re
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
# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["prompt"])
sequences = tokenizer.texts_to_sequences(df["prompt"])
word_index = tokenizer.word_index

max_length = max(len(s) for s in sequences)
data = pad_sequences(sequences, maxlen=max_length)

encoder = OneHotEncoder(sparse_output=False)
labels = encoder.fit_transform(df[["rating"]])

subject_encoder = OneHotEncoder(sparse_output=False)
subject_data = subject_encoder.fit_transform(df[["subject"]])
type_encoder = OneHotEncoder(sparse_output=False)
type_data = type_encoder.fit_transform(df[["type"]])

# Combine the 'prompt', 'subject', and 'type' data
X = np.hstack([data, subject_data, type_data])
print(X)

# Determine the input length
input_length = X.shape[1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Load GloVe embeddings
embeddings_index = {}
with open(os.path.join(current_dir, "assets", "glove.6B.100d.txt")) as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        except ValueError:
            print(
                f"Failed to convert embedding values to float for word '{word}', skipping this word."
            )

# Create an embedding matrix
embedding_matrix = np.zeros(
    (len(word_index) + 1, 100)
)  # assuming you're using 300d GloVe embeddings
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define the model
model = Sequential()
model.add(
    Embedding(
        len(word_index) + 1,
        100,
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False,
    )
)
model.add(GRU(512, return_sequences=True))  # Replaced LSTM with GRU
model.add(Dropout(0.5))
model.add(GRU(256, return_sequences=True))  # Replaced LSTM with GRU
model.add(Dropout(0.5))
model.add(GRU(128))  # Replaced LSTM with GRU
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(5, activation="softmax"))  # Assuming you have 5 different ratings

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer="RMSprop",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
