import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


# -------------------------------
# Feature Functions
# -------------------------------

def size_score(length):

    if 80 <= length <= 200:
        return 1.0
    elif length < 80:
        return np.exp(-0.5 * ((length - 80) / 20) ** 2)
    else:
        return np.exp(-0.5 * ((length - 200) / 50) ** 2)


def repeat_density(seq):

    c = 0
    i = 0
    n = len(seq)

    while i < n:

        j = i
        while j < n and seq[j] == seq[i]:
            j += 1

        if j - i >= 3:
            c += (j - i)

        i = j

    return c / n


def extract_features(seq, start, length):

    frag = (seq + seq)[start:start + length]

    at = (frag.count("A") + frag.count("T")) / length
    gc = (frag.count("G") + frag.count("C")) / length

    cpg = frag.count("CG") / length

    rep = repeat_density(frag)

    size = size_score(length)

    return [at, gc, cpg, rep, size]


# -------------------------------
# Fragment Generator
# -------------------------------

def generate_fragments(seq):

    data = []

    n = len(seq)

    for L in range(60, 260, 20):

        for s in range(0, n, 15):

            f = extract_features(seq, s, L)

            data.append(f)

    cols = ["AT", "GC", "CpG", "Repeat", "Size"]

    return pd.DataFrame(data, columns=cols)


# -------------------------------
# Train ML
# -------------------------------

def train_model(df, y):

    X = df.values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LogisticRegression(max_iter=2000)

    model.fit(Xtr, ytr)

    probs = model.predict_proba(Xte)[:, 1]

    fpr, tpr, _ = roc_curve(yte, probs)

    auc_score = auc(fpr, tpr)

    return model, fpr, tpr, auc_score


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config("FRP-ML Tool", layout="wide")

st.title("🧬 mtDNA FRP-ML Analyzer")


st.markdown("Upload mtDNA sequences and disease labels")


# Upload sequences
fasta = st.file_uploader(
    "Upload FASTA file",
    type=["fa", "fasta", "txt"]
)


# Upload labels
labels = st.file_uploader(
    "Upload CSV (column name: label)",
    type=["csv"]
)


run = st.button("Run Analysis")


# -------------------------------
# Execution
# -------------------------------

if run:

    if not fasta or not labels:

        st.error("Upload FASTA + Label CSV")

    else:

        # Read sequence
        raw = fasta.read().decode()

        seq = "".join(
            [l for l in raw.splitlines()
             if not l.startswith(">")]
        ).upper()


        # Read labels
        lab = pd.read_csv(labels)

        if "label" not in lab.columns:

            st.error("CSV must have 'label' column (0/1)")

        else:

            y = lab["label"].values


            # Feature Extraction
            df = generate_fragments(seq)


            if len(df) != len(y):

                st.error("Label count ≠ fragment count")

            else:

                # Train ML
                model, fpr, tpr, auc_val = train_model(df, y)


                st.success("Training Complete")


                col1, col2 = st.columns(2)

                col1.metric("Fragments", len(df))
                col2.metric("AUC", f"{auc_val:.3f}")


                # ROC Plot
                fig, ax = plt.subplots()

                ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
                ax.plot([0, 1], [0, 1], "--")

                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")

                ax.legend()

                st.pyplot(fig)


                # Feature Weights
                st.subheader("Model Weights")

                weights = pd.DataFrame({
                    "Feature": df.columns,
                    "Weight": model.coef_[0]
                })

                st.dataframe(weights)


                # Download model
                csv = weights.to_csv(index=False).encode()

                st.download_button(
                    "Download Weights",
                    csv,
                    "model_weights.csv"
                )
