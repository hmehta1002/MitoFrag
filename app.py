import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


# =================================
# FRP Feature Functions
# =================================

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


# =================================
# FRP Core Analysis
# =================================

def frp_analysis(sequence, dloop=(16024, 576)):

    n = len(sequence)

    ext = sequence + sequence

    rows = []

    ds, de = dloop


    for L in range(60, 260, 20):

        for s in range(0, n, 15):

            frag = ext[s:s + L]


            at = (frag.count("A") + frag.count("T")) / L
            gc = (frag.count("G") + frag.count("C")) / L

            instab = 1 - gc

            rep = repeat_density(frag)

            cpg = frag.count("CG") / L

            size = size_score(L)


            # D-loop overlap
            is_dloop = False

            for p in range(s, s + L):

                pos = p % n

                if ds < de:

                    if ds <= pos <= de:
                        is_dloop = True
                else:

                    if pos >= ds or pos <= de:
                        is_dloop = True


            raw_fps = (
                0.3 * at +
                0.3 * instab +
                0.2 * rep +
                0.2 * int(is_dloop)
            )

            raw_ivs = (
                0.6 * cpg +
                0.4 * size
            )


            rows.append([
                raw_fps,
                raw_ivs
            ])


    df = pd.DataFrame(
        rows,
        columns=["raw_fps", "raw_ivs"]
    )


    eps = 1e-9


    df["FPS"] = (df.raw_fps - df.raw_fps.min()) / \
                (df.raw_fps.max() - df.raw_fps.min() + eps)

    df["IVS"] = (df.raw_ivs - df.raw_ivs.min()) / \
                (df.raw_ivs.max() - df.raw_ivs.min() + eps)


    df["Alarm"] = df["FPS"] + df["IVS"]

    df["Alarm_norm"] = (df.Alarm - df.Alarm.min()) / \
                       (df.Alarm.max() - df.Alarm.min() + eps)


    return df


# =================================
# Build Patient Feature Vector
# =================================

def patient_profile(sequence):

    df = frp_analysis(sequence)

    features = [
        df["FPS"].mean(),
        df["IVS"].mean(),
        df["Alarm_norm"].mean(),
        df["Alarm_norm"].max(),
        df["Alarm_norm"].std()
    ]

    return features


# =================================
# ML Training
# =================================

def train_model(X, y):

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )


    model = LogisticRegression(max_iter=3000)

    model.fit(Xtr, ytr)


    probs = model.predict_proba(Xte)[:, 1]


    fpr, tpr, _ = roc_curve(yte, probs)

    auc_val = auc(fpr, tpr)


    return model, fpr, tpr, auc_val


# =================================
# Streamlit UI
# =================================

st.set_page_config("FRP Clinical Analyzer", layout="wide")

st.title("🧬 FRP mtDNA Disease Prediction System")


st.markdown("Upload patient FASTA files + labels CSV")


fasta_files = st.file_uploader(
    "Upload Patient FASTA Files",
    type=["fa", "fasta", "txt"],
    accept_multiple_files=True
)


label_file = st.file_uploader(
    "Upload Labels CSV (sample,label)",
    type=["csv"]
)


run_btn = st.button("Run Clinical Analysis")


# =================================
# Execution
# =================================

if run_btn:

    if not fasta_files or not label_file:

        st.error("Upload FASTA files and label CSV")

    else:

        # Read FASTA files
        sequences = {}

        for file in fasta_files:

            raw = file.read().decode()

            seq = "".join([
                l.strip()
                for l in raw.splitlines()
                if not l.startswith(">")
            ]).upper()

            name = file.name.split(".")[0]

            sequences[name] = seq


        # Read labels
        labels = pd.read_csv(label_file)

        if not {"sample", "label"}.issubset(labels.columns):

            st.error("CSV must have: sample,label")

        else:

            X = []
            y = []


            with st.spinner("Running FRP on patients..."):

                for _, row in labels.iterrows():

                    name = row["sample"]
                    label = row["label"]


                    if name not in sequences:

                        st.warning(f"Missing FASTA: {name}")
                        continue


                    seq = sequences[name]

                    feats = patient_profile(seq)

                    X.append(feats)
                    y.append(label)


            X = np.array(X)
            y = np.array(y)


            if len(X) < 5:

                st.error("Need at least 5 samples")

            else:

                # Train ML
                with st.spinner("Training ML model..."):

                    model, fpr, tpr, auc_val = train_model(X, y)


                st.success("Clinical Analysis Complete")


                # Metrics
                col1, col2 = st.columns(2)

                col1.metric("Patients", len(X))
                col2.metric("AUC", f"{auc_val:.3f}")


                # ROC
                st.subheader("ROC Curve")

                fig, ax = plt.subplots()

                ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
                ax.plot([0, 1], [0, 1], "--")

                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")

                ax.set_title("Disease Prediction ROC")

                ax.legend()

                st.pyplot(fig)


                # Feature Table
                st.subheader("Patient-Level Features")

                feat_df = pd.DataFrame(
                    X,
                    columns=[
                        "FPS_mean",
                        "IVS_mean",
                        "Alarm_mean",
                        "Alarm_max",
                        "Alarm_std"
                    ]
                )

                feat_df["Label"] = y

                st.dataframe(feat_df)


st.caption("FRP Clinical Model | Research Prototype")
