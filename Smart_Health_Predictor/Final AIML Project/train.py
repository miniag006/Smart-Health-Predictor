
import argparse, os, json, pickle
import pandas as pd, numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report



def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['prognosis'], errors='ignore').astype(float)
    y = df['prognosis'].astype(str)
    return X, y


def train_one_vs_rest(X_train, y_train):
    classes = sorted(y_train.unique())
    models = {}
    for cls in classes:
        print(f"Training Bayesian model for {cls}...")
        y_bin = (y_train == cls).astype(float)
        br = BayesianRidge(compute_score=True)
        br.fit(X_train, y_bin)
        models[cls] = br
    return models, classes


def get_bayesian_scores(models, X):
    classes = list(models.keys())
    scores = np.column_stack([models[c].predict(X) for c in classes])
    return scores, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/prepared_train.csv")
    parser.add_argument("--test", default="data/prepared_test.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X_train, y_train = load_data(args.train)
    X_test, y_test = load_data(args.test)

    print("Applying StandardScaler normalization...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    import joblib
    scaler_path = os.path.join(args.out, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")


    print("\nTraining Bayesian one-vs-rest models...")
    bayesian_models, classes = train_one_vs_rest(X_train, y_train)

    print("\nGenerating Bayesian features for Random Forest...")
    scores_train, _ = get_bayesian_scores(bayesian_models, X_train)
    scores_test, _ = get_bayesian_scores(bayesian_models, X_test)


    from sklearn.preprocessing import MinMaxScaler
    score_scaler = MinMaxScaler()
    scores_train = score_scaler.fit_transform(scores_train)
    scores_test = score_scaler.transform(scores_test)


    # Train Random Forest on Bayesian outputs
    print("\nTraining RandomForest meta-classifier...")
    from sklearn.calibration import CalibratedClassifierCV

# Train Random Forest and wrap it with probability calibration
    rf_base = RandomForestClassifier(n_estimators=200, random_state=42)
    rf = CalibratedClassifierCV(rf_base, method="sigmoid", cv=3)
    rf.fit(scores_train, y_train)


    # Evaluate hybrid model
    y_pred = rf.predict(scores_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔥 Hybrid Model Accuracy on Test Set: {acc*100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    print("\nSaving models...")
    with open(os.path.join(args.out, "models_bayesian.joblib"), "wb") as f:
        pickle.dump(bayesian_models, f)
    with open(os.path.join(args.out, "random_forest_meta.joblib"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(args.out, "classes_order.json"), "w") as f:
        json.dump(classes, f)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"accuracy": float(acc)}, f, indent=2)

    print("\n✅ Training complete. Saved all artifacts in", args.out)
    with open(os.path.join(args.out, "score_scaler.joblib"), "wb") as f:
        pickle.dump(score_scaler, f)
