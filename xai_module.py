from pathlib import Path

import joblib
import pandas as pd
import shap
import numpy as np

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "intrusion_model.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)


def explain_record(record):
    """
    Explain a single network record.

    record must be a DataFrame with the same
    41 input feature columns used during training.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform the input using the trained preprocessing pipeline
    transformed = preprocessor.transform(record)

    # SHAP works with dense arrays for this explanation
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=float)

    # Get the feature names after encoding
    feature_names = preprocessor.get_feature_names_out()

    # Explain the Random Forest prediction
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    # Find the class index for Attack
    attack_index = list(classifier.classes_).index("Attack")

    # Handle SHAP output formats across versions
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[attack_index])[0]
    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[0, :, attack_index]
        else:
            values = values[0]

    # Return the five features with the largest
    # absolute contribution to the attack prediction
    ranking = np.argsort(np.abs(values))[::-1][:5]

    explanation = [
        {
            "feature": str(feature_names[i]),
            "contribution": round(float(values[i]), 4)
        }
        for i in ranking
    ]

    prediction = model.predict(record)[0]

    return {
        "prediction": prediction,
        "top_features": explanation
    }