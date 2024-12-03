import os
import re
import nltk
import spacy
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import textstat
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from transformers import BertTokenizer, BertModel
from xgboost import XGBClassifier


class SentimentAnalyzer:
    def __init__(self):
        self.current_dir = os.getcwd()
        self.export_dir = os.path.join(self.current_dir, "export")
        self.setup_nlp()

    def setup_nlp(self):
        nltk.download("punkt")
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

    def clean_text(self, text):
        if not isinstance(text, str):
            return text
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"\W", " ", text)
        text = re.sub(r"\s+", " ", text, flags=re.I)
        return text.lower()

    def bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    class TextFeatures(BaseEstimator, TransformerMixin):
        def __init__(self, analyzer):
            self.analyzer = analyzer

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            df = pd.DataFrame(X)
            df["prompt_length"] = df["prompt"].apply(len)
            df["unique_words"] = df["prompt"].apply(
                lambda x: len(set(nltk.word_tokenize(x)))
            )
            df["num_entities"] = df["prompt"].apply(
                lambda x: len(self.analyzer.nlp(x).ents)
            )
            df["sentiment"] = df["prompt"].apply(
                lambda x: TextBlob(x).sentiment.polarity
            )
            df["readability"] = df["prompt"].apply(
                textstat.flesch_reading_ease)
            df["bert_embedding"] = df["prompt"].apply(
                self.analyzer.bert_embedding)

            df.to_csv(os.path.join(self.analyzer.export_dir, "text_features.csv"),
                      index=False)
            return df.drop(columns=["prompt"])

    def prepare_data(self):
        df = pd.read_csv(os.path.join(self.export_dir, "general.csv"))

        for col in ["prompt", "response", "subject", "type"]:
            df[col] = df[col].apply(self.clean_text)

        df["group"] = (df["prompt"].str.strip().str.lower()
                       != "continue").cumsum()
        df_grouped = df[df["prompt"].str.strip().str.lower() != "continue"].groupby(
            "group").first().reset_index()
        df_grouped["rating"] = df.groupby("group")["rating"].median()
        df_grouped["rating_class"] = df_grouped["rating"].apply(
            lambda x: "positive" if x > 0 else "negative")

        df_grouped.to_csv(os.path.join(
            self.export_dir, "grouped_data.csv"), index=False)
        return df_grouped.fillna({"prompt": ""})

    def create_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", TfidfVectorizer(stop_words="english"), "prompt"),
                ("cat", OneHotEncoder(handle_unknown="ignore"),
                 ["subject", "type"]),
                ("text_features", Pipeline([
                    ("generator", self.TextFeatures(self)),
                    ("scaler", StandardScaler())
                ]), ["prompt"]),
            ]
        )

        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    def train_model(self):
        df_grouped = self.prepare_data()
        X = df_grouped[["prompt", "subject", "type"]]
        y = df_grouped["rating_class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        pipeline = self.create_pipeline()
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.1, 1],
            "model__max_depth": [3, 5, 7],
        }

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)
        self.plot_results(y_test, y_pred)

        return grid_search.best_params_, accuracy_score(y_test, y_pred)

    def plot_results(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.savefig(os.path.join(self.export_dir, "confusion_matrix.png"))
        plt.close()


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    best_params, accuracy = analyzer.train_model()
    print(f"Best parameters: {best_params}")
    print(f"Accuracy: {accuracy}")
