import argparse
import os
import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(args):
    # =========================
    # Load dataset
    # =========================
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["Personality"])
    y = df["Personality"]

    # =========================
    # Split data
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # =========================
    # Train model
    # =========================
    with mlflow.start_run(run_name="GradientBoosting_CI"):
        model = GradientBoostingClassifier(
            random_state=args.random_state
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # =========================
        # Metrics
        # =========================
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # =========================
        # Params
        # =========================
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_samples", X_train.shape[0])

        # =========================
        # Save model
        # =========================
        output_dir = os.path.dirname(args.model_output)
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, args.model_output)

        print("✅ Training selesai")
        print("📦 Model disimpan di:", args.model_output)
        print("📂 Absolute path:", os.path.abspath(args.model_output))
        print("📁 Isi folder output:", os.listdir(output_dir))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI Training for Personality Classification")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    main(args)
