import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import json
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, ConfusionMatrixDisplay
)

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=200)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--min_samples_split", type=int, default=2)
parser.add_argument("--data_path", type=str, default="telco_preprocessing/telco_preprocessed.csv")
args = parser.parse_args()

RANDOM_STATE = 42
ARTIFACT_DIR = "artifacts_tmp"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv(args.data_path)
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ─── MLflow ───────────────────────────────────────────────────────────────────
with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Log params
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("random_state", RANDOM_STATE)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    roc_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    mlflow.log_artifact(roc_path, artifact_path="plots")

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    rep_path = os.path.join(ARTIFACT_DIR, "classification_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(rep_path, artifact_path="reports")

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

print("[INFO] Training selesai")
