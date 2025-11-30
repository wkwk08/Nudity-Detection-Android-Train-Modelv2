import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import paths from config.py
from config import OUTPUT_DIR, LOGS_DIR

st.set_page_config(page_title="Nudity Detection Dashboard", layout="wide")
st.title("📊 Nudity Detection Training & Evaluation Dashboard")

OBJECTIVES = ["Objective_1", "Objective_2", "Objective_3", "Objective_4"]

# === TABS FOR NAVIGATION ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Dataset Summary",
    "📊 Evaluation Metrics",
    "📈 Training Logs",
    "📊 Class Distribution",
    "📈 Confusion Matrices"
])

# === TAB 1: DATASET SUMMARY ===
with tab1:
    st.subheader("📁 Dataset Split Summary")
    try:
        summary_path = os.path.join(LOGS_DIR, "validation_totals.csv")
        df = pd.read_csv(summary_path)
        df.columns = ["Objective", "Total"]
        df = df[df["Objective"] != "GrandTotal"]
        df["Total"] = df["Total"].astype(int)
        st.dataframe(df)

        grand_total = pd.read_csv(summary_path)
        grand = grand_total[grand_total["Objective"] == "GrandTotal"]["TotalCount"].values[0]
        st.markdown(f"**📦 Grand Total Images Across All Objectives: {grand}**")
    except FileNotFoundError:
        st.dataframe(pd.DataFrame(columns=["Objective", "Total"]))

# === TAB 2: FINAL EVALUATION METRICS ===
with tab2:
    try:
        eval_path = os.path.join(LOGS_DIR, "evaluation_results.csv")
        eval_df = pd.read_csv(eval_path)
        st.subheader("Final Evaluation Metrics (ML vs Baseline)")
        st.dataframe(eval_df)

        for metric in ["Acc", "Prec", "Rec", "F1"]:
            with st.expander(f"{metric} Comparison"):
                fig, ax = plt.subplots()
                ax.bar(eval_df["Objective"], eval_df[f"ML_{metric}"], width=0.4, label="ML Model", align="center")
                ax.bar(eval_df["Objective"], eval_df[f"Base_{metric}"], width=0.4, label="Baseline", align="edge")
                ax.set_title(f"ML vs Baseline ({metric})")
                ax.set_ylabel(metric)
                ax.set_ylim(0, 1.1)
                ax.legend()
                st.pyplot(fig)

        avg_metrics = eval_df[["ML_Acc", "ML_Prec", "ML_Rec", "ML_F1"]].mean()
        st.markdown("### 📊 Aggregate ML Performance Across All Objectives")
        st.write(avg_metrics)
    except FileNotFoundError:
        st.dataframe(pd.DataFrame(columns=["Objective", "ML_Acc", "Base_Acc", "ML_Prec", "Base_Prec", "ML_Rec", "Base_Rec", "ML_F1", "Base_F1"]))

# === TAB 3: TRAINING LOGS ===
with tab3:
    for obj in OBJECTIVES:
        with st.expander(f"Training Results for {obj}", expanded=False):
            log_file = os.path.join(LOGS_DIR, f"{obj}_training_log.csv")
            try:
                df = pd.read_csv(log_file)
                st.subheader("Training Metrics Table")
                st.dataframe(df)

                fig_acc, ax_acc = plt.subplots()
                ax_acc.plot(df["Epoch"], df["TrainAcc"], label="Train Accuracy", marker="o")
                ax_acc.plot(df["Epoch"], df["ValAcc"], label="Validation Accuracy", marker="o")
                if "TestAcc" in df.columns:
                    ax_acc.plot(df["Epoch"], df["TestAcc"], label="Test Accuracy", marker="o")
                ax_acc.set_xlabel("Epoch")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.legend()
                st.pyplot(fig_acc)

                fig_loss, ax_loss = plt.subplots()
                ax_loss.plot(df["Epoch"], df["TrainLoss"], label="Train Loss", marker="o", color="red")
                if "ValLoss" in df.columns:
                    ax_loss.plot(df["Epoch"], df["ValLoss"], label="Validation Loss", marker="o", color="orange")
                ax_loss.set_xlabel("Epoch")
                ax_loss.set_ylabel("Loss")
                ax_loss.legend()
                st.pyplot(fig_loss)
            except FileNotFoundError:
                st.dataframe(pd.DataFrame(columns=["Epoch", "TrainAcc", "ValAcc", "TestAcc", "TrainLoss", "ValLoss"]))

# === TAB 4: CLASS DISTRIBUTION ===
with tab4:
    st.subheader("Class Distribution per Objective")
    try:
        dist_path = os.path.join(LOGS_DIR, "validation_summary.csv")
        dist_df = pd.read_csv(dist_path)
        st.dataframe(dist_df)

        for obj in OBJECTIVES:
            obj_df = dist_df[dist_df["Objective"] == obj]
            fig, ax = plt.subplots()
            grouped = obj_df.groupby(["Split", "Class"])["Count"].sum().unstack().fillna(0)
            grouped.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"Class Distribution for {obj}")
            ax.set_ylabel("Image Count")
            st.pyplot(fig)
    except FileNotFoundError:
        st.dataframe(pd.DataFrame(columns=["Objective", "Dataset", "Split", "Class", "Count"]))

# === TAB 5: CONFUSION MATRICES ===
with tab5:
    st.subheader("Confusion Matrices per Objective")
    for obj in OBJECTIVES:
        try:
            cm_path = os.path.join(LOGS_DIR, f"{obj}_confusion.csv")
            cm_df = pd.read_csv(cm_path, index_col=0)
            st.write(f"Confusion Matrix for {obj}")
            fig, ax = plt.subplots()
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        except FileNotFoundError:
            st.dataframe(pd.DataFrame(columns=["Actual", "Predicted"]))