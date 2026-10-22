from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

# Standard NSL-KDD column names
columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Check that both files exist
if not TRAIN_FILE.exists() or not TEST_FILE.exists():
    raise FileNotFoundError(
        "Dataset files not found. Check the dataset folder."
    )

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE, header=None, names=columns)
test_df = pd.read_csv(TEST_FILE, header=None, names=columns)

# Remove missing and duplicate records
train_df = train_df.dropna().drop_duplicates()
test_df = test_df.dropna().drop_duplicates()

# Convert attack labels into two classes:
# normal = Normal, all attack types = Attack
y_train = (
    train_df["label"].astype(str).str.strip().str.lower()
    .eq("normal").map({True: "Normal", False: "Attack"})
)

y_test = (
    test_df["label"].astype(str).str.strip().str.lower()
    .eq("normal").map({True: "Normal", False: "Attack"})
)

# Use the 41 network features only.
# Exclude both label and difficulty.
X_train = train_df.drop(columns=["label", "difficulty"])
X_test = test_df.drop(columns=["label", "difficulty"])

# Convert categorical features to numerical indicators.
categorical_features = ["protocol_type", "service", "flag"]
numerical_features = [
    col for col in X_train.columns
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numerical", "passthrough", numerical_features)
    ]
)


# Build the model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

print("Training Random Forest model...")
model.fit(X_train, y_train)

# Evaluate on the separate test dataset
predictions = model.predict(X_test)

print("\n========== MODEL EVALUATION ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(
    y_test, predictions, labels=["Normal", "Attack"]
))

# Save the complete pipeline
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / "intrusion_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved at:", model_path)