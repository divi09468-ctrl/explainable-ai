
from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "intrusion_model.joblib"
TEST_PATH = BASE_DIR / "dataset" / "KDDTest+.txt"

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

print("Loading trained model...")
model = joblib.load(MODEL_PATH)

print("Loading test dataset...")
df = pd.read_csv(
    TEST_PATH,
    header=None,
    names=columns
)

X_test = df.drop(columns=["label", "difficulty"])

y_test = df["label"].apply(
    lambda label: "Normal"
    if str(label).strip().lower() == "normal"
    else "Attack"
)

print("Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["Normal", "Attack"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print("Class order: Normal, Attack")
print(confusion_matrix(y_test, y_pred, labels=["Normal", "Attack"]))

print("\nEvaluation completed successfully!")
