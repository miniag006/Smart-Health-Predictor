
import argparse, os, json, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def evaluate_model(models_dir, test_path):
 
    bayesian_path = os.path.join(models_dir, "models_bayesian.joblib")
    rf_path = os.path.join(models_dir, "random_forest_meta.joblib")
    classes_path = os.path.join(models_dir, "classes_order.json")

    if not (os.path.exists(bayesian_path) and os.path.exists(rf_path) and os.path.exists(classes_path)):
        raise FileNotFoundError("Models or metadata not found. Please run train.py first.")

    print("🔹 Loading trained models...")
    bayesian_models = joblib.load(bayesian_path)
    rf = joblib.load(rf_path)
    with open(classes_path, "r") as f:
        classes = json.load(f)

    
    print("🔹 Loading test data...")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["prognosis"], errors="ignore")
    y_test = test_df["prognosis"]

   
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        from joblib import load
        scaler = load(scaler_path)
        print("🔹 Applying saved StandardScaler normalization...")
        X_test = scaler.transform(X_test)
    else:
        print("⚠️ Warning: scaler.joblib not found. Using raw test data.")


    print("🔹 Generating Bayesian feature predictions...")
    scores = np.column_stack([bayesian_models[c].predict(X_test) for c in classes])


    print("🔹 Predicting final diseases using RandomForest meta-classifier...")
    y_pred = rf.predict(scores)

   
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    
    out_dir = models_dir
    os.makedirs(out_dir, exist_ok=True)

    report = classification_report(y_test, y_pred, output_dict=True)
    with open(os.path.join(out_dir, "eval_report.json"), "w") as f:
        json.dump({"accuracy": float(acc), "report": report}, f, indent=2)

    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix — Accuracy: {acc*100:.2f}%", fontsize=16)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.show()

    print(f"\n📊 Confusion matrix saved as: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models", help="Path to model folder")
    parser.add_argument("--test", default="data/prepared_test.csv", help="Path to test CSV file")
    args = parser.parse_args()

    evaluate_model(args.models, args.test)
