import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


# --------------------------------
# Feature Functions
# --------------------------------

def size_score(length):

    if 80 <= length <= 200:
        return 1.0
    elif length < 80:
        return np.exp(-0.5 * ((length - 80) / 20) ** 2)
    else:
        return np.exp(-0.5 * ((length - 200) / 50) ** 2)


def repeat_density(seq):

    count = 0
    i = 0
    n = len(seq)

    while i < n:

        j = i

        while j < n and seq[j] == seq[i]:
            j += 1

        if j - i >= 3:
            count += (j - i)

        i = j

    return count / n


def extract_features(seq, start, length):

    frag = (seq + seq)[start:start + length]

    at = (frag.count("A") + frag.count("T")) / length
    gc = (frag.count("G") + frag.count("C")) / length

    cpg = frag.count("CG") / length

    rep = repeat_density(frag)

    size = size_score(length)

    return [at, gc, cpg, rep, size]


# --------------------------------
# Fragment Generator
# --------------------------------

def generate_fragments(seq):

    rows = []

    n = len(seq)

    for L in range(60, 260, 20):

        for s in range(0, n, 15):

            feat = extract_features(seq, s, L)

            rows.append(feat)

    cols = ["AT", "GC", "CpG", "Repeat", "Size"]

    return pd.DataFrame(rows, columns=cols)


# --------------------------------
# ML Training
# --------------------------------

def train_model(df, y):

    X = df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )

    model = LogisticRegression(max_iter=3000)

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probs)

    auc_score = auc(fpr, tpr)

    return model, fpr, tpr, auc_score


# --------------------------------
# Streamlit UI
# --------------------------------

st.set_page_config(
    page_title="FRP-ML Analyzer",
    layout="wide"
)

st.title("🧬 mtDNA FRP-ML Analyzer")

st.write("Upload mtDNA sequence and label file (optional)")


fasta_file = st.file_uploader(
    "Upload FASTA File",
    type=["fa", "fasta", "txt"]
)


label_file = st.file_uploader(
    "Upload Label CSV (column: label)",
    type=["csv"]
)


run_btn = st.button("Run Analysis")


# --------------------------------
# Execution
# --------------------------------

if run_btn:

    if not fasta_file:

        st.error("Please upload a FASTA file")

    else:

        # Read FASTA
        raw = fasta_file.read().decode()

        seq = "".join([
            l.strip()
            for l in raw.splitlines()
            if not l.startswith(">")
        ]).upper()


        # Generate fragments
        df = generate_fragments(seq)


        # Read labels if provided
        if label_file:

            lab = pd.read_csv(label_file)

            if "label" in lab.columns:

                y = lab["label"].values

            else:

                st.warning("No 'label' column. Using random labels.")

                y = np.random.randint(0, 2, len(df))

        else:

            st.warning("No label file. Using random labels.")

            y = np.random.randint(0, 2, len(df))


        # Fix size mismatch
        if len(y) != len(df):

            st.warning("Label count mismatch. Auto-adjusting.")

            y = np.random.randint(0, 2, len(df))


        # Train model
        with st.spinner("Training ML model..."):

            model, fpr, tpr, auc_val = train_model(df, y)


        st.success("Analysis Complete")


        # Metrics
        col1, col2 = st.columns(2)

        col1.metric("Fragments", len(df))
        col2.metric("AUC Score", f"{auc_val:.3f}")


        # ROC Plot
        st.subheader("ROC Curve")

        fig, ax = plt.subplots()

        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")

        ax.plot([0, 1], [0, 1], "--")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.set_title("ROC Curve")

        ax.legend()

        st.pyplot(fig)


        # Feature Weights
        st.subheader("Feature Importance")

        weights = pd.DataFrame({
            "Feature": df.columns,
            "Weight": model.coef_[0]
        })

        st.dataframe(weights)


        # Download
        csv = weights.to_csv(index=False).encode()

        st.download_button(
            "Download Model Weights",
            csv,
            "model_weights.csv",
            "text/csv"
        )


st.markdown("---")
st.caption("FRP-ML Prototype | Himani Project")
