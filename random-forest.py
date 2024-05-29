import pandas as pd
import mplcatppuccin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from metaCost import MetaCost
from matplotlib.patches import FancyBboxPatch
from scipy.stats import mode
import joblib
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.style.use("mocha")


def load_and_preprocess_data():
    current_dir = os.getcwd()
    df = pd.read_csv(os.path.join(current_dir, "export", "processed_data.csv"))

    le = LabelEncoder()
    categorical_features = ["subject", "type"]

    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])

    df["createdAt"] = pd.to_datetime(df["createdAt"])

    df["createdAt"] = (df["createdAt"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(
        "1s"
    )

    df["rating"] = df["rating"].apply(lambda x: 1 if x > 0 else 0)

    grouped = df.groupby("group").agg(
        {
            "prompt": "first",
            "response": lambda x: (
                x.value_counts().index[0] if not x.value_counts().empty else None
            ),
            "subject": lambda x: (
                x.value_counts().index[0] if not x.value_counts().empty else None
            ),
            "type": lambda x: (
                x.value_counts().index[0] if not x.value_counts().empty else None
            ),
            "failed": "mean",
            "createdAt": "mean",
            "canceled": "mean",
            "rating": "mean",
            "is_continue": "mean",
        }
    )

    vectorizer = TfidfVectorizer()

    grouped["prompt"] = grouped["prompt"].fillna("")

    # Fit and transform the 'prompt' column
    X_prompt = vectorizer.fit_transform(grouped["prompt"])

    # Convert the result to a DataFrame
    X_prompt_df = pd.DataFrame(X_prompt.toarray())

    # Set the columns of the DataFrame
    X_prompt_df.columns = vectorizer.get_feature_names_out()

    # Add the DataFrame to the original DataFrame with a suffix for overlapping column names
    grouped = grouped.join(X_prompt_df, rsuffix="_prompt")

    # Drop the original 'prompt' column
    grouped = grouped.drop(columns=["prompt"])

    # Round the mean ratings to the nearest integer
    grouped["rating"] = grouped["rating"].round().astype(int)

    return grouped


def select_features(df):

    # Split the data into training and testing sets
    y = df["rating"]

    X = df.drop("rating", axis=1)

    # Get the numerical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns

    # Fill NA values in numerical columns with their mean
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

    # Select only the numerical columns
    X = X.select_dtypes(include=[np.number])

    # Apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # Concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ["Specs", "Score"]  # Naming the dataframe columns

    # Print 10 best features
    print(featureScores.nlargest(10, "Score"))

    # Create a new DataFrame with only the selected features
    top_10_features = featureScores.nlargest(10, "Score")["Specs"].values

    # Select only the top 10 features from your data
    X_selected = X[top_10_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    models["Random Forest"].param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    models["SVM"].param_grid = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["rbf"],
    }

    models["Gradient Boosting"].param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.1, 0.05, 0.02, 0.01],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [20, 50, 100, 150],
    }

    models["Naive Bayes"].param_grid = {
        "var_smoothing": np.logspace(0, -9, num=100),
    }

    for model in models:
        grid_search = GridSearchCV(
            estimator=models[model],
            param_grid=models[model].param_grid,
            cv=10,
            n_jobs=-1,
            verbose=2,
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        models[model].set_params(**best_params)

        models[model].fit(X_train, y_train)

    return models


def evaluate_models(models, X_test, y_test):
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"{model}\n")
        print(classification_report(y_test, y_pred))

        joblib.dump(model, f'{model_name.replace(" ", "_").lower()}.joblib')


def plot_results(models, X_test, y_test):
    current_dir = os.getcwd()
    # Initialize the figure
    plt.figure(figsize=(10, 15))

    for i, (model_name, model) in enumerate(models.items()):
        # Compute confusion matrix
        cm = confusion_matrix(y_test, model.predict(X_test))

        # Normalize confusion matrix
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Subplot
        ax = plt.subplot(len(models), 1, i + 1)
        sns.heatmap(
            cm, annot=True, fmt=".2f", linewidths=0.5, square=True, cmap="Blues_r"
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        all_sample_title = f"{model_name} Accuracy Score: {model.score(X_test, y_test)}"
        ax.set_title(all_sample_title, size=15, loc="center")  # Center the title

        new_patches = []
        for patch in reversed(ax.patches):
            bb = patch.get_bbox()
            color = patch.get_facecolor()
            p_bbox = FancyBboxPatch(
                (bb.xmin, bb.ymin),
                abs(bb.width),
                abs(bb.height),
                boxstyle="round,pad=-0,rounding_size=0.015",
                ec="none",
                fc=color,
            )
            patch.remove()
            new_patches.append(p_bbox)
        for patch in new_patches:
            ax.add_patch(patch)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                "plots",
                f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png',
            ),
            dpi=300,
        )

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "plots", "confusion_matrix.png"))

    # Bar chart comparison
    accuracies = [model.score(X_test, y_test) for model_name, model in models]

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    bars = ax.bar(models.keys(), accuracies)

    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch(
            (bb.xmin, bb.ymin),
            abs(bb.width),
            abs(bb.height),
            boxstyle="round,pad=-0.0140,rounding_size=0.015",
            ec="none",
            fc=color,
        )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)

    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.savefig(os.path.join(current_dir, "plots", "model_accuracy_comparison.png"))


def main():
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = select_features(df)
    models = train_models(X_train, y_train)
    print(models)
    evaluate_models(models, X_test, y_test)
    plot_results(models, X_test, y_test)


if __name__ == "__main__":
    main()


# # Load the data
# current_dir = os.getcwd()
# df = pd.read_csv(os.path.join(current_dir, "export", "processed_data.csv"))

# # Select only numeric columns from the DataFrame
# numeric_cols = df.select_dtypes(include=[np.number])

# # Compute the correlation matrix
# corrmat = numeric_cols.corr()

# top_corr_features = corrmat.index
# plt.figure(figsize=(20, 20))

# # Plot heat map
# g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
# plt.savefig("heatmap.png")


# # Preprocess the data
# # Convert categorical variables into numerical ones
# le = LabelEncoder()
# categorical_features = ["subject", "type"]
# for feature in categorical_features:
#     df[feature] = le.fit_transform(df[feature])

# # Convert 'createdAt' to datetime
# df["createdAt"] = pd.to_datetime(df["createdAt"])

# # Convert 'createdAt' to the number of seconds since the Unix epoch
# df["createdAt"] = (df["createdAt"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

# # Convert 'rating' to binary
# df["rating"] = df["rating"].apply(lambda x: 1 if x > 0 else 0)

# # Group by 'group' and aggregate
# grouped = df.groupby("group").agg(
#     {
#         "prompt": "first",
#         "response": lambda x: (
#             x.value_counts().index[0] if not x.value_counts().empty else None
#         ),
#         "subject": lambda x: (
#             x.value_counts().index[0] if not x.value_counts().empty else None
#         ),
#         "type": lambda x: (
#             x.value_counts().index[0] if not x.value_counts().empty else None
#         ),
#         "failed": "mean",
#         "createdAt": "mean",
#         "canceled": "mean",
#         "rating": "mean",
#         "is_continue": "mean",
#     }
# )

# vectorizer = TfidfVectorizer()

# grouped["prompt"] = grouped["prompt"].fillna("")

# # Fit and transform the 'prompt' column
# X_prompt = vectorizer.fit_transform(grouped["prompt"])

# # Convert the result to a DataFrame
# X_prompt_df = pd.DataFrame(X_prompt.toarray())

# # Set the columns of the DataFrame
# X_prompt_df.columns = vectorizer.get_feature_names_out()

# # Add the DataFrame to the original DataFrame with a suffix for overlapping column names
# grouped = grouped.join(X_prompt_df, rsuffix="_prompt")

# # Drop the original 'prompt' column
# grouped = grouped.drop(columns=["prompt"])

# # Round the mean ratings to the nearest integer
# print(grouped)

# # Split the data into training and testing sets
# # Split the data into training and testing sets
# X = grouped.drop("rating", axis=1)

# # Get the numerical columns
# numerical_cols = X.select_dtypes(include=[np.number]).columns

# y = grouped["rating"]

# # Fill NA values in numerical columns with their mean
# X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

# # Select only the numerical columns
# X = X.select_dtypes(include=[np.number])


# # Define the parameter grid
# param_grid = {
#     "n_estimators": [100, 200, 300, 400, 500],
#     "max_depth": [None, 10, 20, 30, 40, 50],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "bootstrap": [True, False],
# }

# # Create a base model
# rf = RandomForestClassifier(random_state=42)

# # Instantiate the grid search model
# grid_search = GridSearchCV(
#     estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
# )

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Get the best parameters
# best_params_rdc = grid_search.best_params_

# # Train a new classifier using the best parameters from the grid search
# clf_rdc = RandomForestClassifier(**best_params_rdc)
# clf_rdc.fit(X_train, y_train)

# # Evaluate the new classifier
# y_pred_rdc = clf_rdc.predict(X_test)


# # Calculatge class frequencies
# class_freq = y_train.value_counts(normalize=True)

# # Calculate the inverse of class freq
# inverse_class_freq = 1 / class_freq

# # Create a dictionary with the class weights
# C = np.zeros((len(class_freq), len(class_freq)))

# np.fill_diagonal(C, inverse_class_freq)

# # Create a DataFrame from your training data
# S = pd.DataFrame(X_train)
# S["target"] = y_train

# # Create a MetaCost instance
# metacost = MetaCost(S, clf_rdc, C)

# # Fit the MetaCost model
# model = metacost.fit("target", len(class_freq))

# # Now you can use the model to make predictions
# y_pred_metacost_rdf = model.predict(X_test)


# from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier

# # Define the parameter grid for SVM
# param_grid_svm = {
#     "C": [0.1, 1, 10, 100, 1000],
#     "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
#     "kernel": ["rbf"],
# }

# # Create a base model for SVM
# svm = SVC()

# # Instantiate the grid search model for SVM
# grid_search_svm = GridSearchCV(
#     estimator=svm, param_grid=param_grid_svm, cv=3, n_jobs=-1, verbose=2
# )

# # Fit the grid search to the data
# grid_search_svm.fit(X_train, y_train)

# # Get the best parameters
# best_params_svm = grid_search_svm.best_params_

# # Train a new classifier using the best parameters from the grid search
# clf_svm = SVC(**best_params_svm)
# clf_svm.fit(X_train, y_train)

# # Evaluate the new classifier
# y_pred_svm = clf_svm.predict(X_test)

# # Define the parameter grid for Gradient Boosting
# param_grid_gb = {
#     "n_estimators": [100, 200, 300, 400, 500],
#     "learning_rate": [0.1, 0.05, 0.02, 0.01],
#     "max_depth": [4, 6, 8],
#     "min_samples_leaf": [20, 50, 100, 150],
# }

# # Create a base model for Gradient Boosting
# gb = GradientBoostingClassifier()

# # Instantiate the grid search model for Gradient Boosting
# grid_search_gb = GridSearchCV(
#     estimator=gb, param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2
# )

# # Fit the grid search to the data
# grid_search_gb.fit(X_train, y_train)

# # Get the best parameters
# best_params_gb = grid_search_gb.best_params_

# # Train a new classifier using the best parameters from the grid search
# clf_gb = GradientBoostingClassifier(**best_params_gb)
# clf_gb.fit(X_train, y_train)

# # Evaluate the new classifier
# y_pred_gb = clf_gb.predict(X_test)


# # Naive Bayes
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred_gnb = gnb.predict(X_test)


# Print the classification report
# print("METACOST MODEL WITH RDC\n")
# print(classification_report(y_test, y_pred_metacost_rdf))

# print("NAIVE BAYES \n")
# print(classification_report(y_test, y_pred_gnb))

# print("GRADIENT BOOSTING \n")
# print(best_params_gb)
# print(classification_report(y_test, y_pred_gb))

# print("SVM \n")
# print(best_params_svm)
# print(classification_report(y_test, y_pred_svm))

# print("RDC \n")
# print(best_params_rdc)
# print(classification_report(y_test, y_pred_rdc))

# from sklearn.metrics import confusion_matrix

# import matplotlib as mpl
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# mpl.style.use("mocha")
# import seaborn as sns

# # List of models
# models = [clf_rdc, clf_svm, clf_gb, gnb, model]
# model_names = ["Random Forest", "SVM", "Gradient Boosting", "Naive Bayes", "MetaCost"]

# # Initialize the figure
# plt.figure(figsize=(10, 15))

# from matplotlib.patches import FancyBboxPatch

# for i, model in enumerate(models):
#     # Compute confusion matrix
#     cm = confusion_matrix(y_test, model.predict(X_test))

#     # Normalize confusion matrix
#     cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

#     # Subplot
#     ax = plt.subplot(len(models), 1, i + 1)
#     sns.heatmap(cm, annot=True, fmt=".2f", linewidths=0.5, square=True, cmap="Blues_r")
#     plt.ylabel("Actual label")
#     plt.xlabel("Predicted label")
#     all_sample_title = f"{model_names[i]} Accuracy Score: {model.score(X_test, y_test)}"
#     ax.set_title(all_sample_title, size=15, loc="center")  # Center the title

#     new_patches = []
#     for patch in reversed(ax.patches):
#         bb = patch.get_bbox()
#         color = patch.get_facecolor()
#         p_bbox = FancyBboxPatch(
#             (bb.xmin, bb.ymin),
#             abs(bb.width),
#             abs(bb.height),
#             boxstyle="round,pad=-0,rounding_size=0.015",
#             ec="none",
#             fc=color,
#         )
#         patch.remove()
#         new_patches.append(p_bbox)
#     for patch in new_patches:
#         ax.add_patch(patch)

#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(
#             "export",
#             f'confusion_matrix_{model_names[i].replace(" ", "_").lower()}.png',
#         ),
#         dpi=300,
#     )

# # Show the plot
# plt.tight_layout()
# plt.savefig(os.path.join(current_dir, "plots", "confusion_matrix.png"))

# # Bar chart comparison
# accuracies = [model.score(X_test, y_test) for model in models]

# plt.figure(figsize=(10, 5))
# ax = plt.gca()
# bars = ax.bar(model_names, accuracies)

# new_patches = []
# for patch in reversed(ax.patches):
#     bb = patch.get_bbox()
#     color = patch.get_facecolor()
#     p_bbox = FancyBboxPatch(
#         (bb.xmin, bb.ymin),
#         abs(bb.width),
#         abs(bb.height),
#         boxstyle="round,pad=-0.0140,rounding_size=0.015",
#         ec="none",
#         fc=color,
#     )
#     patch.remove()
#     new_patches.append(p_bbox)
# for patch in new_patches:
#     ax.add_patch(patch)

# plt.xlabel("Models")
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy Comparison")
# plt.savefig(os.path.join(current_dir, "plots", "model_accuracy_comparison.png"))
