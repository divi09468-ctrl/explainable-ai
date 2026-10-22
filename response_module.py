from datetime import datetime


def calculate_risk(attack_probability, attack_type="unknown",
                   asset_criticality=0.5):

    # Keep inputs within the range 0 to 1
    p = max(0.0, min(float(attack_probability), 1.0))
    a = max(0.0, min(float(asset_criticality), 1.0))

    # Example risk weights for this prototype
    attack_weights = {
        "normal": 0.0,
        "dos": 0.9,
        "probe": 0.6,
        "r2l": 0.8,
        "u2r": 1.0,
        "unknown": 0.5
    }

    attack_key = str(attack_type).strip().lower()
    w = attack_weights.get(attack_key, attack_weights["unknown"])

    risk_score = (0.5 * p) + (0.3 * w) + (0.2 * a)
    risk_score = round(risk_score, 4)

    if risk_score < 0.40:
        severity = "Low"
        action = "Log and monitor"
    elif risk_score < 0.70:
        severity = "Medium"
        action = "Alert administrator; review before action"
    else:
        severity = "High"
        action = "Alert administrator; recommend simulated blocking"

    return {
        "risk_score": risk_score,
        "severity": severity,
        "recommended_action": action
    }


# Application-level simulation only.
# This does not modify the real computer firewall.
simulated_blocklist = set()


def simulate_block(ip_address):
    ip_address = str(ip_address).strip()

    if not ip_address:
        return {"status": "error", "message": "IP address is required"}

    simulated_blocklist.add(ip_address)

    return {
        "status": "simulated",
        "ip_address": ip_address,
        "message": "Added to the application's simulated blocklist",
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }


def get_simulated_blocklist():
    return sorted(simulated_blocklist)