"""
Fake News Detection System - Web UI (Streamlit)

Run:  streamlit run app.py
Needs fake_news_detection.py in the same folder.
"""
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

import fake_news_detection as fnd

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

EXAMPLES = {
    "Sample: likely REAL": (
        "The finance ministry said on Tuesday that the government will present its budget "
        "next month, with officials expecting moderate economic growth and a lower fiscal "
        "deficit, according to a statement released after the committee meeting."),
    "Sample: likely FAKE": (
        "SHOCKING: Secret documents EXPOSED! You won't believe what the media is hiding "
        "from you about this miracle cure the government doesn't want you to know about!!!"),
}


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_model(mtime):
    """mtime is only used so the cache refreshes after retraining."""
    return joblib.load(fnd.MODEL_PATH)


def model_mtime():
    return fnd.MODEL_PATH.stat().st_mtime if fnd.MODEL_PATH.exists() else None


def fake_probability(pipe, text):
    """Return P(fake) in 0-1 for any of the models."""
    s = fnd.scores(pipe, [text])[0]
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        return float(s)
    return float(1 / (1 + np.exp(-s)))      # SVM: squash decision score


# --------------------------------------------------------------- page: check
def page_check():
    st.header("🔍 Check a news article")
    mt = model_mtime()
    if mt is None:
        st.warning("No trained model found yet. Go to **Train Models** in the sidebar first.")
        return
    pipe = load_model(mt)
    st.caption(f"Model in use: **{type(pipe.named_steps['clf']).__name__}**")

    choice = st.selectbox("Try an example (optional)", ["—"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    text = st.text_area("Paste a headline or full article", value=default, height=220)

    if st.button("Analyze", type="primary"):
        clean = fnd.clean_text(text)
        if len(clean.split()) < 3:
            st.error("Please enter a longer text (at least a few meaningful words).")
            return
        p_fake = fake_probability(pipe, clean)
        label_fake = p_fake >= 0.5
        conf = p_fake if label_fake else 1 - p_fake

        c1, c2 = st.columns([1, 2])
        with c1:
            if label_fake:
                st.error("🚨 Likely FAKE")
            else:
                st.success("✅ Likely REAL")
            st.metric("Confidence", f"{conf * 100:.1f}%")
        with c2:
            st.write("Probability of being fake")
            st.progress(min(max(p_fake, 0.0), 1.0))
            st.caption("This is a statistical guess based on writing style and word "
                       "patterns, not a fact-check. Always verify with trusted sources.")

        # which words pushed the decision (only for linear models)
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "coef_"):
            vec = pipe.named_steps["tfidf"]
            X = vec.transform([clean])
            feats = np.array(vec.get_feature_names_out())
            contrib = X.toarray().ravel() * clf.coef_.ravel()
            nz = np.nonzero(contrib)[0]
            if len(nz):
                top = nz[np.argsort(-np.abs(contrib[nz]))[:10]]
                st.subheader("Words that influenced the result")
                st.dataframe(pd.DataFrame({
                    "word / phrase": feats[top],
                    "pushes toward": ["FAKE" if contrib[i] > 0 else "REAL" for i in top],
                    "strength": np.round(np.abs(contrib[top]), 3),
                }), hide_index=True)


# --------------------------------------------------------------- page: train
def page_train():
    st.header("⚙️ Train models")
    st.write("Upload the Kaggle files, or leave empty to use `data/Fake.csv` and `data/True.csv`.")

    c1, c2 = st.columns(2)
    fake_file = c1.file_uploader("Fake.csv", type="csv")
    real_file = c2.file_uploader("True.csv", type="csv")
    sample = st.slider("Number of articles to use (smaller = faster)", 1000, 45000, 8000, step=1000)
    test_size = st.slider("Test split", 0.1, 0.4, 0.2, step=0.05)

    all_models = list(fnd.get_models())
    chosen = st.multiselect("Models", all_models, default=all_models)

    if not st.button("Train and evaluate", type="primary"):
        return
    if not chosen:
        st.error("Select at least one model.")
        return

    try:
        df = fnd.load_data(fake_file or "data/Fake.csv", real_file or "data/True.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Upload both CSV files or place them in a `data/` folder.")
        return

    fnd.OUT_DIR.mkdir(exist_ok=True)
    df = df.sample(n=min(sample, len(df)), random_state=42)
    with st.spinner("Cleaning text..."):
        df["clean_text"] = df["content"].apply(fnd.clean_text)
        df = df[df["clean_text"].str.len() > 0]
    st.info(f"{len(df)} articles | fake: {(df.label == 1).sum()} | real: {(df.label == 0).sum()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean_text"], df["label"], test_size=test_size, stratify=df["label"], random_state=42)

    rows, roc_data, cms, fitted = {}, {}, {}, {}
    bar = st.progress(0.0)
    status = st.empty()
    models = {k: v for k, v in fnd.get_models().items() if k in chosen}
    for i, (name, clf) in enumerate(models.items()):
        status.write(f"Training **{name}**...")
        t0 = time.time()
        pipe = fnd.make_pipeline(clf).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        s = fnd.scores(pipe, X_te)
        fpr, tpr, _ = roc_curve(y_te, s)
        auc = roc_auc_score(y_te, s)
        rows[name] = {"Accuracy": accuracy_score(y_te, pred),
                      "Precision": precision_score(y_te, pred),
                      "Recall": recall_score(y_te, pred),
                      "F1 Score": f1_score(y_te, pred),
                      "ROC AUC": auc,
                      "Train time (s)": time.time() - t0}
        roc_data[name] = (fpr, tpr, auc)
        cms[name] = confusion_matrix(y_te, pred)
        fitted[name] = pipe
        bar.progress((i + 1) / len(models))
    status.empty()

    results = pd.DataFrame(rows).T.round(4)
    results.to_csv(fnd.OUT_DIR / "results.csv")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    fnd.plot_results(results[metric_cols], roc_data, cms)
    if "Logistic Regression" in fitted:
        fnd.plot_top_features(fitted["Logistic Regression"])

    best = results["F1 Score"].idxmax()
    joblib.dump(fitted[best], fnd.MODEL_PATH)
    st.success(f"Done! Best model: **{best}** (saved). Open **Results** or **Check News** in the sidebar.")
    st.dataframe(results)


# -------------------------------------------------------------- page: results
def page_results():
    st.header("📊 Results")
    csv = fnd.OUT_DIR / "results.csv"
    if not csv.exists():
        st.warning("No results yet. Train the models first.")
        return
    results = pd.read_csv(csv, index_col=0)
    st.subheader("Metrics table")
    st.dataframe(results.style.highlight_max(axis=0, color="#cfe8cf"))
    st.download_button("Download results.csv", csv.read_bytes(), "results.csv")

    for title, fname in [("Model comparison", "model_comparison.png"),
                         ("ROC curves", "roc_curves.png"),
                         ("Confusion matrices", "confusion_matrices.png"),
                         ("Most influential words (Logistic Regression)", "top_features.png")]:
        f = fnd.OUT_DIR / fname
        if f.exists():
            st.subheader(title)
            st.image(str(f))


# ----------------------------------------------------------------------- main
st.title("📰 Fake News Detection System")
st.caption("TF-IDF + machine learning (Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost)")

page = st.sidebar.radio("Menu", ["Check News", "Train Models", "Results"])
{"Check News": page_check, "Train Models": page_train, "Results": page_results}[page]()