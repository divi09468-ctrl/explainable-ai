from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from xai_module import explain_record, model
from response_module import (
    calculate_risk,
    simulate_block,
    get_simulated_blocklist
)
app = Flask(__name__)

BASE_DIR = Path(__file__).parent
TEST_FILE = BASE_DIR / "dataset" / "KDDTest+.txt"

COLUMNS = [
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

# Load a small set of test records for the demonstration UI.
df = pd.read_csv(TEST_FILE, header=None, names=COLUMNS)
features = df.drop(columns=["label", "difficulty"])
total_records = len(df)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            row_number = int(request.form.get("row_number", "0"))
            asset_criticality = float(
                request.form.get("asset_criticality", "0.5")
            )

            if not 0 <= row_number < len(features):
                raise ValueError(
                    f"Choose a row from 0 to {len(features) - 1}."
                )

            if not 0 <= asset_criticality <= 1:
                raise ValueError(
                    "Asset criticality must be between 0 and 1."
                )

            record = features.iloc[[row_number]]
            prediction = model.predict(record)[0]
            probabilities = model.predict_proba(record)[0]
            classes = list(model.named_steps["classifier"].classes_)
            attack_probability = float(
                probabilities[classes.index("Attack")]
            )

            explanation = explain_record(record)

            risk = calculate_risk(
                attack_probability=attack_probability,
                attack_type="unknown" if prediction == "Attack" else "normal",
                asset_criticality=asset_criticality
            )

            result = {
                "row_number": row_number,
                "prediction": prediction,
                "attack_probability": round(attack_probability * 100, 2),
                "risk_score": risk["risk_score"],
                "severity": risk["severity"],
                "recommended_action": risk["recommended_action"],
                "top_features": explanation["top_features"]
            }

        except Exception as exc:
            error = str(exc)

    return render_template(
    "index.html",
    result=result,
    error=error,
    total_records=total_records,
    blocklist=get_simulated_blocklist()
)


@app.route("/simulate-block", methods=["POST"])
def simulate_block_route():
    ip_address = request.form.get("ip_address", "").strip()

    if ip_address:
        simulate_block(ip_address)

    return render_template(
        "index.html",
        total_records=total_records,
        blocklist=get_simulated_blocklist()
    )

@app.route("/evaluation")
def evaluation():
    # Convert actual labels into Normal or Attack
    y_test = df["label"].apply(
        lambda label: "Normal"
        if str(label).strip().lower() == "normal"
        else "Attack"
    )

    # Predict using the currently loaded model
    y_pred = model.predict(features)

    # Calculate evaluation metrics
    metrics = {
        "accuracy": round(
            accuracy_score(y_test, y_pred) * 100, 2
        ),
        "precision": round(
            precision_score(
                y_test, y_pred,
                pos_label="Attack", zero_division=0
            ) * 100, 2
        ),
        "recall": round(
            recall_score(
                y_test, y_pred,
                pos_label="Attack", zero_division=0
            ) * 100, 2
        ),
        "f1": round(
            f1_score(
                y_test, y_pred,
                pos_label="Attack", zero_division=0
            ) * 100, 2
        )
    }

    # Confusion matrix: Normal first, Attack second
    cm = confusion_matrix(
        y_test, y_pred,
        labels=["Normal", "Attack"]
    )

    return render_template(
        "evaluation.html",
        metrics=metrics,
        cm=cm.tolist()
    )


if __name__ == "__main__":
    app.run(debug=True)