import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


# --------------------------------
# FRP Feature Functions
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


# --------------------------------
# FRP Analysis (Immune Alarm Core)
# --------------------------------

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


            # Raw Scores
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
                s,
                (s + L) % n,
                L,
                raw_fps,
                raw_ivs
            ])


    df = pd.DataFrame(
        rows,
        columns=[
            "start", "end", "length",
            "raw_fps", "raw_ivs"
        ]
    )


    eps = 1e-9


    # Normalize
    df["FPS"] = (df.raw_fps - df.raw_fps.min()) / \
                (df.raw_fps.max() - df.raw_fps.min() + eps)

    df["IVS"] = (df.raw_ivs - df.raw_ivs.min()) / \
                (df.raw_ivs.max() - df.raw_ivs.min() + eps)


    # Final Alarm Score
    df["Alarm"] = df["FPS"] + df["IVS"]

    df["Alarm_norm"] = (df.Alarm - df.Alarm.min()) / \
                       (df.Alarm.max() - df.Alarm.min() + eps)


    return df


# --------------------------------
# Alarm Landscape Plot
# --------------------------------

def plot_alarm(df, n, dloop):

    base = np.zeros(n)


    for _, r in df.iterrows():

        s = int(r.start)
        e = int(r.end)

        score = r.Alarm_norm


        if s < e:

            base[s:e] = np.maximum(base[s:e], score)

        else:

            base[s:] = np.maximum(base[s:], score)
            base[:e] = np.maximum(base[:e], score)


    fig, ax = plt.subplots(figsize=(14, 5))


    ax.plot(base, color="crimson")

    ax.fill_between(range(n), base, color="crimson", alpha=0.15)


    ds, de = dloop


    if ds > de:

        ax.axvspan(ds, n, color="gray", alpha=0.2)
        ax.axvspan(0, de, color="gray", alpha=0.2)

    else:

        ax.axvspan(ds, de, color="gray", alpha=0.2)


    ax.set_title("mtDNA Immune Alarm Landscape (FRP)")

    ax.set_xlabel("Position (bp)")

    ax.set_ylabel("Normalized Risk")


    return fig


# --------------------------------
# ML Training
# --------------------------------

def train_model(df, y):

    X = df[["FPS", "IVS", "Alarm_norm"]].values


    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )


    model = LogisticRegression(max_iter=3000)

    model.fit(Xtr, ytr)


    prob = model.predict_proba(Xte)[:, 1]


    fpr, tpr, _ = roc_curve(yte, prob)

    auc_val = auc(fpr, tpr)


    return model, fpr, tpr, auc_val


# --------------------------------
# Streamlit UI
# --------------------------------

st.set_page_config("FRP Immune Alarm Tool", layout="wide")

st.title("🧬 mtDNA FRP Immune Alarm Analyzer")


fasta = st.file_uploader(
    "Upload FASTA",
    type=["fa", "fasta", "txt"]
)


run = st.button("Run FRP Analysis")



# --------------------------------
# Execution
# --------------------------------

if run:

    if not fasta:

        st.error("Upload FASTA file")

    else:

        raw = fasta.read().decode()

        seq = "".join([
            l.strip()
            for l in raw.splitlines()
            if not l.startswith(">")
        ]).upper()


        # FRP
        with st.spinner("Running FRP model..."):

            df = frp_analysis(seq)


        st.success("FRP Analysis Complete")


        # Alarm Plot
        st.subheader("Immune Alarm Landscape")

        fig = plot_alarm(df, len(seq), (16024, 576))

        st.pyplot(fig)


        # Auto labels
        y = np.random.randint(0, 2, len(df))


        # ML
        with st.spinner("Training ML classifier..."):

            model, fpr, tpr, auc_val = train_model(df, y)


        col1, col2 = st.columns(2)

        col1.metric("Fragments", len(df))

        col2.metric("AUC", f"{auc_val:.3f}")


        # ROC
        st.subheader("ROC Curve")

        fig2, ax2 = plt.subplots()

        ax2.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")

        ax2.plot([0, 1], [0, 1], "--")

        ax2.legend()

        st.pyplot(fig2)


        # Table
        st.subheader("Top Alarm Fragments")

        st.dataframe(
            df.sort_values("Alarm_norm", ascending=False).head(30)
        )


st.caption("FRP Immune Alarm Model | Research Prototype")
