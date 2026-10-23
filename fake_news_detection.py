"""
Fake News Detection System
--------------------------
TF-IDF (unigrams + bigrams) + Logistic Regression, Decision Tree, Random Forest,
Linear SVM and XGBoost. Produces metrics table, confusion matrices, ROC curves,
top-feature plot, and saves the best model for predictions.

Usage:
    # Train + evaluate (Kaggle dataset: Fake.csv and True.csv)
    python fake_news_detection.py --fake data/Fake.csv --real data/True.csv

    # Predict with the saved model
    python fake_news_detection.py --predict "Some news article text here"
"""
import argparse
import re
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # XGBoost is optional
    XGBClassifier = None

OUT_DIR = Path("outputs")
MODEL_PATH = OUT_DIR / "best_model.joblib"

# ---------------------------------------------------------------- stop words
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set(ENGLISH_STOP_WORDS)  # offline fallback


# ------------------------------------------------------------- preprocessing
def clean_text(text: str) -> str:
    """Remove HTML/URLs/punctuation/digits, lowercase, drop stop words."""
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)                 # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()     # letters only, lowercase
    return " ".join(w for w in text.split() if w not in STOP_WORDS and len(w) > 2)


def load_data(fake_path, real_path) -> pd.DataFrame:
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake["label"], real["label"] = 1, 0                # 1 = fake, 0 = real

    # The Kaggle real-news articles often start with "CITY (Reuters) -", which
    # lets models "cheat". Strip that source tag so they learn real content.
    real["text"] = real["text"].astype(str).str.replace(
        r"^.{0,60}?\(Reuters\)\s*-?\s*", "", regex=True)

    df = pd.concat([fake, real], ignore_index=True)
    title = df["title"].fillna("") if "title" in df else ""
    df["content"] = (title + " " + df["text"].fillna("")).str.strip()
    df = df[df["content"].str.len() > 0].drop_duplicates(subset="content")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# -------------------------------------------------------------------- models
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Decision Tree": DecisionTreeClassifier(max_depth=30, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "SVM (Linear)": LinearSVC(random_state=42),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                          n_jobs=-1, eval_metric="logloss", random_state=42)
    return models


def make_pipeline(clf):
    # Vectorizer lives inside the pipeline so it is fit on training data only
    # (no data leakage) and stays sparse (no .toarray() memory blowup).
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                  min_df=3, max_df=0.9, sublinear_tf=True)),
        ("clf", clf),
    ])


def scores(pipe, X):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


# ---------------------------------------------------------------------- plots
def plot_results(results, roc_data, cms):
    names = list(results.index)

    ax = results[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]].plot(
        kind="bar", figsize=(10, 5), ylim=(0.8, 1.01), rot=15)
    ax.set_title("Model comparison")
    plt.tight_layout(); plt.savefig(OUT_DIR / "model_comparison.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for n, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{n} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(OUT_DIR / "roc_curves.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        ax.imshow(cms[n], cmap="Blues")
        ax.set_title(n, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cms[n][i, j], ha="center", va="center",
                        color="white" if cms[n][i, j] > cms[n].max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=150); plt.close()


def plot_top_features(pipe, k=15):
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return
    feats = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = clf.coef_.ravel()
    idx = np.r_[np.argsort(coef)[:k], np.argsort(coef)[-k:]]
    plt.figure(figsize=(8, 7))
    plt.barh(feats[idx], coef[idx], color=["tab:green"] * k + ["tab:red"] * k)
    plt.title("Top words: green = real news, red = fake news")
    plt.tight_layout(); plt.savefig(OUT_DIR / "top_features.png", dpi=150); plt.close()


# ----------------------------------------------------------------------- main
def train(args):
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading data...")
    df = load_data(args.fake, args.real)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
    print(f"{len(df)} articles | fake: {(df.label == 1).sum()} | real: {(df.label == 0).sum()}")

    print("Cleaning text...")
    df["clean_text"] = df["content"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

    rows, roc_data, cms, fitted = {}, {}, {}, {}
    for name, clf in get_models().items():
        t0 = time.time()
        pipe = make_pipeline(clf).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        s = scores(pipe, X_test)
        fpr, tpr, _ = roc_curve(y_test, s)
        auc = roc_auc_score(y_test, s)
        rows[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "ROC AUC": auc,
        }
        roc_data[name] = (fpr, tpr, auc)
        cms[name] = confusion_matrix(y_test, pred)
        fitted[name] = pipe
        print(f"  {name:<20} acc={rows[name]['Accuracy']:.4f}  f1={rows[name]['F1 Score']:.4f}  ({time.time()-t0:.0f}s)")

    results = pd.DataFrame(rows).T.round(4)
    print("\n=== Results ===\n", results.to_string())
    results.to_csv(OUT_DIR / "results.csv")

    plot_results(results, roc_data, cms)
    plot_top_features(fitted["Logistic Regression"])

    best = results["F1 Score"].idxmax()
    joblib.dump(fitted[best], MODEL_PATH)
    print(f"\nBest model: {best} -> saved to {MODEL_PATH}")
    print(f"Plots and results.csv saved in ./{OUT_DIR}/")


def predict(text):
    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found. Train first (see --help).")
    pipe = joblib.load(MODEL_PATH)
    clean = clean_text(text)
    label = pipe.predict([clean])[0]
    s = scores(pipe, [clean])[0]
    print(f"Prediction: {'FAKE' if label == 1 else 'REAL'}  (score={s:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fake News Detection System")
    p.add_argument("--fake", default="data/Fake.csv", help="path to fake news CSV")
    p.add_argument("--real", default="data/True.csv", help="path to real news CSV")
    p.add_argument("--sample", type=int, default=0, help="use only N articles (fast test)")
    p.add_argument("--predict", type=str, help="classify this text with the saved model")
    a = p.parse_args()
    predict(a.predict) if a.predict else train(a)