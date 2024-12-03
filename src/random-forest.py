import os

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcatppuccin
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from metaCost import MetaCost

mpl.style.use("mocha")


def load_and_preprocess_data():
    current_dir = os.getcwd()
    df = pd.read_csv(os.path.join(current_dir, "export", "processed_data.csv"))

    le = LabelEncoder()
    categorical_features = ["subject", "type"]
    label_mappings = {}

    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])
        label_mappings[feature] = dict(
            zip(le.transform(le.classes_), le.classes_))

    df["createdAt"] = pd.to_datetime(df["createdAt"])

    df["createdAt"] = (df["createdAt"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(
        "1s"
    )

    df["rating"] = df["rating"].apply(lambda x: 1 if x > 0 else 0)

    grouped = df.groupby("group").agg(
        {
            "prompt": "first",
            "response": lambda x: (
                x.value_counts().index[0] if not x.value_counts(
                ).empty else None
            ),
            "subject": lambda x: (
                x.value_counts().index[0] if not x.value_counts(
                ).empty else None
            ),
            "type": lambda x: (
                x.value_counts().index[0] if not x.value_counts(
                ).empty else None
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

    return grouped, label_mappings


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
    joblibs_dir = os.path.join(os.getcwd(), "joblibs")
    os.makedirs(joblibs_dir, exist_ok=True)

    model_files = {
        "Random Forest": "random_forest.joblib",
        "SVM": "svm.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "Naive Bayes": "naive_bayes.joblib",
    }

    # Check for existing models
    existing_models = {}
    models_to_train = {}

    for model_name, filename in model_files.items():
        # Check in root directory
        filepath = os.path.join(os.getcwd(), filename)
        if os.path.exists(filepath):
            print(f"Loading existing model: {model_name}")
            existing_models[model_name] = joblib.load(filepath)
        else:
            print(f"Model not found, will train: {model_name}")
            if model_name == "Random Forest":
                models_to_train[model_name] = RandomForestClassifier(
                    random_state=42)
            elif model_name == "SVM":
                models_to_train[model_name] = SVC()
            elif model_name == "Gradient Boosting":
                models_to_train[model_name] = GradientBoostingClassifier()
            elif model_name == "Naive Bayes":
                models_to_train[model_name] = GaussianNB()

    # Define parameter grids for models that need training
    if "Random Forest" in models_to_train:
        models_to_train["Random Forest"].param_grid = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }

    if "SVM" in models_to_train:
        models_to_train["SVM"].param_grid = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
            "kernel": ["rbf"],
        }

    if "Gradient Boosting" in models_to_train:
        models_to_train["Gradient Boosting"].param_grid = {
            "n_estimators": [100, 200, 300, 400, 500],
            "learning_rate": [0.1, 0.05, 0.02, 0.01],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [20, 50, 100, 150],
        }

    if "Naive Bayes" in models_to_train:
        models_to_train["Naive Bayes"].param_grid = {
            "var_smoothing": np.logspace(0, -9, num=100),
        }

    # Train and save models that don't exist
    for model_name, model in models_to_train.items():
        print(f"Training {model_name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=model.param_grid,
            cv=10,
            n_jobs=-1,
            verbose=2,
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        model.set_params(**best_params)
        model.fit(X_train, y_train)

        # Save the newly trained model
        save_path = os.path.join(os.getcwd(), model_files[model_name])
        joblib.dump(model, save_path)
        print(f"Saved {model_name} to {save_path}")

        existing_models[model_name] = model

    return existing_models


def evaluate_models(models, X_test, y_test):
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"{model}\n")
        print(classification_report(y_test, y_pred))

        joblib.dump(model, f'{model_name.replace(" ", "_").lower()}.joblib')


def plot_results(models, X_test, y_test):
    current_dir = os.getcwd()

    # Initialize the figure for confusion matrices
    plt.figure(figsize=(10, 15))

    for i, (model_name, model) in enumerate(models.items()):
        # Compute confusion matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        # Normalize confusion matrix
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Subplot with adjusted position
        ax = plt.subplot(len(models), 1, i + 1)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",  # Format to 2 decimal places
            linewidths=0.5,
            square=True,
            cmap="Blues_r",
            center=0.5,  # Center the color mapping
        )

        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")

        # Format accuracy score to 2 decimal places
        accuracy = model.score(X_test, y_test)
        all_sample_title = f"{model_name} Accuracy Score: {accuracy:.2f}"

        # Center the title and adjust padding
        ax.set_title(all_sample_title, size=15, pad=20)

        # Rounded corners with centered position
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

    # Adjust layout and save confusion matrices
    # Add padding around the plots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(
        os.path.join(current_dir, "plots", "confusion_matrix.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # Bar chart comparison with centered plots and 2 decimal formatting
    accuracies = [
        float(f"{model.score(X_test, y_test):.2f}")
        for model_name, model in models.items()
    ]
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    bars = ax.bar(list(models.keys()), accuracies)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # Rounded corners for bars
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

    # Center the bar plot and save
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(
        os.path.join(current_dir, "plots", "model_accuracy_comparison.png"),
        bbox_inches="tight",
        dpi=300,
    )


def plot_best_models_per_category(
    models, X_test, y_test, df, label_mappings, category="subject"
):
    current_dir = os.getcwd()

    # Convert y_test to numpy array if it's a pandas Series
    y_test_arr = y_test.to_numpy() if hasattr(
        y_test, "to_numpy") else np.array(y_test)

    # Get the indices of the test set
    if hasattr(X_test, "index"):
        test_indices = X_test.index
    else:
        test_indices = np.arange(len(y_test_arr))

    # Create a dataframe with only test data
    df_test = df.loc[test_indices].copy()
    df_test = df_test.reset_index(drop=True)

    # Get unique categories
    unique_categories = df_test[category].unique()

    # Calculate per-category metrics for each model
    model_performances = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        model_performances[model_name] = {
            "predictions": y_pred,
            "overall_accuracy": model.score(X_test, y_test_arr),
        }

        # Calculate per-category accuracy
        for cat in unique_categories:
            cat_mask = df_test[category].values == cat
            if np.any(cat_mask):
                cat_indices = np.where(cat_mask)[0]
                if len(cat_indices) > 0:
                    # Make sure indices are within bounds
                    cat_indices = cat_indices[cat_indices < len(y_test_arr)]
                    if len(cat_indices) > 0:
                        cat_y_test = y_test_arr[cat_indices]
                        cat_y_pred = y_pred[cat_indices]
                        cat_accuracy = np.mean(cat_y_test == cat_y_pred)
                        model_performances[model_name][f"category_{
                            cat}"] = cat_accuracy

    # Find best models for each category
    category_best_models = {}
    for cat in unique_categories:
        cat_scores = {
            model_name: perf[f"category_{cat}"]
            for model_name, perf in model_performances.items()
            if f"category_{cat}" in perf
        }
        # Get top 2 models for this category
        best_models = sorted(cat_scores.items(),
                             key=lambda x: x[1], reverse=True)[:2]
        category_best_models[cat] = best_models

    # Plot confusion matrices for best models per category
    num_categories = len(unique_categories)
    plt.figure(figsize=(15, 5 * num_categories))

    for cat_idx, (cat, best_models) in enumerate(category_best_models.items()):
        for model_idx, (model_name, score) in enumerate(best_models):
            # Calculate position in subplot grid
            plot_idx = cat_idx * 2 + model_idx + 1

            # Get confusion matrix for this category
            cat_mask = df_test[category].values == cat
            cat_indices = np.where(cat_mask)[0]

            if len(cat_indices) > 0:
                # Make sure indices are within bounds
                cat_indices = cat_indices[cat_indices < len(y_test_arr)]
                if len(cat_indices) > 0:
                    y_test_cat = y_test_arr[cat_indices]
                    y_pred_cat = model_performances[model_name]["predictions"][
                        cat_indices
                    ]

                    cm = confusion_matrix(y_test_cat, y_pred_cat)
                    cm_normalized = cm.astype(
                        "float") / cm.sum(axis=1)[:, np.newaxis]

                    # Create subplot
                    ax = plt.subplot(num_categories, 2, plot_idx)
                    sns.heatmap(
                        cm_normalized,
                        annot=True,
                        fmt=".2f",
                        linewidths=0.5,
                        square=True,
                        cmap="Blues_r",
                        center=0.5,
                    )

                    plt.ylabel("Actual label")
                    plt.xlabel("Predicted label")

                    # Add title with model name and metrics
                    original_cat_name = label_mappings[category][cat].title()
                    title = f"Best Model #{model_idx + 1} for {category.capitalize()}: {original_cat_name}\n{model_name}\nCategory Acc: {
                        score:.2f}, Overall Acc: {model_performances[model_name]['overall_accuracy']:.2f}"
                    ax.set_title(title, size=12, pad=20)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(
        os.path.join(current_dir, "plots", f"best_models_per_{category}.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # Create bar plot comparing the best models
    num_plots = (
        len(unique_categories) + 7
    ) // 4  # Calculate the number of plots needed
    for plot_num in range(num_plots):
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        # Prepare data for bar plot
        all_best_models = list(
            set(
                [
                    model
                    for best_models in category_best_models.values()
                    for model, _ in best_models
                ]
            )
        )

        # Set up bar positions
        x = np.arange(len(all_best_models))
        width = 0.8 / (min(6, len(unique_categories) - plot_num * 6) + 1)

        # Plot overall accuracy
        bars = []
        bars.append(
            ax.bar(
                x - width *
                (min(6, len(unique_categories) - plot_num * 6) / 2),
                [model_performances[m]["overall_accuracy"]
                    for m in all_best_models],
                width,
                label="Overall",
                alpha=0.8,
            )
        )

        # Plot category-specific accuracies
        for i, cat in enumerate(unique_categories[plot_num * 6: (plot_num + 1) * 6]):
            category_accs = []
            for model in all_best_models:
                acc = model_performances[model].get(f"category_{cat}", 0)
                category_accs.append(acc)
            bars.append(
                ax.bar(
                    x
                    - width
                    * (min(6, len(unique_categories) - plot_num * 6) / 2 - (i + 1)),
                    category_accs,
                    width,
                    label=f"{label_mappings[category][cat].title()}",
                    alpha=0.8,
                )
            )

        # Add labels and formatting
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.title(f"Model Performance by {category.capitalize()}")
        plt.xticks(x, all_best_models, rotation=45)

        # Update legend labels to title case
        handles, labels = ax.get_legend_handles_labels()
        labels = [label.title() for label in labels]
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add value labels on the bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        for bar_group in bars:
            autolabel(bar_group)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                current_dir,
                "plots",
                f"best_models_{category}_comparison_{plot_num + 1}.png",
            ),
            bbox_inches="tight",
            dpi=300,
        )


def analyze_all_categories(models, X_test, y_test, df):
    # Add other categorical features as needed
    categories = ["subject", "type"]
    for category in categories:
        plot_best_models_per_category(models, X_test, y_test, df, category)


def plot_best_models_per_class(models, X_test, y_test):
    current_dir = os.getcwd()

    # Calculate per-class metrics for each model
    model_performances = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        model_performances[model_name] = report

    # Find best models for each class
    class_best_models = {}
    for class_label in ["0", "1"]:  # Binary classification
        class_scores = {
            model_name: perf[class_label]["f1-score"]
            for model_name, perf in model_performances.items()
        }
        # Get top 2 models for this class
        best_models = sorted(class_scores.items(),
                             key=lambda x: x[1], reverse=True)[:2]
        class_best_models[class_label] = best_models

    # Plot confusion matrices for best models per class
    plt.figure(figsize=(15, 10))

    for class_idx, (class_label, best_models) in enumerate(class_best_models.items()):
        for model_idx, (model_name, score) in enumerate(best_models):
            # Calculate position in subplot grid
            plot_idx = class_idx * 2 + model_idx + 1

            # Get confusion matrix
            y_pred = models[model_name].predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            # Create subplot
            ax = plt.subplot(2, 2, plot_idx)
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                square=True,
                cmap="Blues_r",
                center=0.5,
            )

            plt.ylabel("Actual label")
            plt.xlabel("Predicted label")

            # Add title with model name and metrics
            accuracy = models[model_name].score(X_test, y_test)
            title = f"Best Model #{model_idx + 1} for Class {class_label}\n{
                model_name}\nAccuracy: {accuracy:.2f}, F1: {score:.2f}"
            ax.set_title(title, size=12, pad=20)

            # Add rounded corners
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

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(
        os.path.join(current_dir, "plots", "best_models_per_class.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # Create bar plot comparing the best models
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Prepare data for bar plot
    all_best_models = list(
        set(
            [
                model
                for best_models in class_best_models.values()
                for model, _ in best_models
            ]
        )
    )
    model_metrics = []

    for model_name in all_best_models:
        accuracy = models[model_name].score(X_test, y_test)
        metrics = {
            "Model": model_name,
            "Accuracy": accuracy,
            "Class 0 F1": model_performances[model_name]["0"]["f1-score"],
            "Class 1 F1": model_performances[model_name]["1"]["f1-score"],
        }
        model_metrics.append(metrics)

    # Plot bars for each metric
    x = np.arange(len(all_best_models))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        [m["Accuracy"] for m in model_metrics],
        width,
        label="Accuracy",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x,
        [m["Class 0 F1"] for m in model_metrics],
        width,
        label="Class 0 F1",
        alpha=0.8,
    )
    bars3 = ax.bar(
        x + width,
        [m["Class 1 F1"] for m in model_metrics],
        width,
        label="Class 1 F1",
        alpha=0.8,
    )

    # Add labels and formatting
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Performance Metrics of Best Models per Class")
    plt.xticks(x, all_best_models, rotation=45)
    plt.legend()

    # Add value labels on the bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(current_dir, "plots",
                     "best_models_metrics_comparison.png"),
        bbox_inches="tight",
        dpi=300,
    )


def main():
    df, label_mappings = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = select_features(df)
    models = train_models(X_train, y_train)
    print(models)
    evaluate_models(models, X_test, y_test)
    plot_results(models, X_test, y_test)
    plot_best_models_per_class(models, X_test, y_test)
    plot_best_models_per_category(models, X_test, y_test, df, label_mappings)


if __name__ == "__main__":
    main()
