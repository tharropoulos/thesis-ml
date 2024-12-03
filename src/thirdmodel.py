import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import re
import os
import nltk
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from transformers import BertTokenizer, BertModel
from textblob import TextBlob
import textstat
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

nltk.download("punkt")
# Load the spacy model for NER
nlp = spacy.load("en_core_web_sm")

current_dir = os.getcwd()
df = pd.read_csv(os.path.join(current_dir, "export", "general.csv"))
stop = stopwords.words("english")


def bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings


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


class TextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a DataFrame from the input Series
        df = pd.DataFrame(X)

        # Create new features
        df["prompt_length"] = df["prompt"].apply(len)
        df["unique_words"] = df["prompt"].apply(
            lambda x: len(set(nltk.word_tokenize(x)))
        )
        df["num_entities"] = df["prompt"].apply(lambda x: len(nlp(x).ents))

        df["sentiment"] = df["prompt"].apply(lambda x: TextBlob(x).sentiment.polarity)
        df["readability"] = df["prompt"].apply(textstat.flesch_reading_ease)
        df["bert_embedding"] = df["prompt"].apply(bert_embedding)

        df.to_csv(os.path.join(current_dir, "export", "text_features.csv"), index=False)
        df = df.drop(columns=["prompt"])

        return df


df["prompt"] = df["prompt"].apply(clean_text)
df["response"] = df["response"].apply(clean_text)
df["subject"] = df["subject"].apply(clean_text)
df["type"] = df["type"].apply(clean_text)

df["group"] = (df["prompt"].str.strip().str.lower() != "continue").cumsum()

# Keep the first non-"continue" prompt in each group
df_grouped = (
    df[df["prompt"].str.strip().str.lower() != "continue"]
    .groupby("group")
    .first()
    .reset_index()
)

# Calculate the median rating for each group
df_grouped["rating"] = df.groupby("group")["rating"].median().reset_index(drop=True)

# Classify the median rating as positive or negative
df_grouped["rating_class"] = df_grouped["rating"].apply(
    lambda x: "positive" if x > 0 else "negative"
)

df_grouped.to_csv(os.path.join(current_dir, "export", "grouped_data.csv"), index=False)
# Define preprocessing steps
nan_rows = df_grouped[
    df_grouped["prompt"].isna()
    | df_grouped["subject"].isna()
    | df_grouped["type"].isna()
]
print(nan_rows)
df_grouped["prompt"] = df_grouped["prompt"].fillna("")

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(stop_words="english"), "prompt"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["subject", "type"]),
        (
            "text_features",
            Pipeline([("generator", TextFeatures()), ("scaler", StandardScaler())]),
            ["prompt"],
        ),
    ]
)

# The rest of your code remains the same

X = df_grouped[["prompt", "subject", "type"]]
y = df_grouped["rating_class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

# Combine preprocessing and modeling steps into a pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Define the hyperparameters to tune and their values
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.1, 1],
    "model__max_depth": [3, 5, 7],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")

grid_search.fit(X_train, y_train)

print("Best parameters: ", grid_search.best_params_)

y_pred = grid_search.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Truth")

plt.savefig(os.path.join(current_dir, "export", "confusion_matrix.png"))
