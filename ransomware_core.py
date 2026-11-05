"""Safe, simulation-based ransomware behavior detection utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

FEATURES = [
    "files_modified", "files_renamed", "files_deleted", "directories_touched",
    "extensions_affected", "ransom_extensions", "events_per_second",
    "high_entropy_files", "entropy_average", "suspicious_process",
    "backup_activity", "network_connections"
]

def generate_sample(ransomware: bool, rng=None) -> dict:
    """Generate a labelled, synthetic behavior sample for classroom demonstration."""
    rng = rng or np.random.default_rng()
    if ransomware:
        modified = int(rng.integers(80, 1001))
        renamed = int(rng.integers(50, max(51, modified + 1)))
        return {
            "files_modified": modified,
            "files_renamed": renamed,
            "files_deleted": int(rng.integers(0, 101)),
            "directories_touched": int(rng.integers(5, 51)),
            "extensions_affected": int(rng.integers(3, 11)),
            "ransom_extensions": int(rng.integers(20, max(21, renamed + 1))),
            "events_per_second": int(rng.integers(30, 501)),
            "high_entropy_files": int(rng.integers(30, max(31, modified + 1))),
            "entropy_average": float(rng.uniform(7.0, 8.0)),
            "suspicious_process": 1,
            "backup_activity": int(rng.integers(0, 2)),
            "network_connections": int(rng.integers(5, 51)),
            "label": 1,
        }
    modified = int(rng.integers(1, 31))
    return {
        "files_modified": modified,
        "files_renamed": int(rng.integers(0, min(5, modified) + 1)),
        "files_deleted": int(rng.integers(0, 4)),
        "directories_touched": int(rng.integers(1, 6)),
        "extensions_affected": int(rng.integers(1, 5)),
        "ransom_extensions": 0,
        "events_per_second": int(rng.integers(1, 11)),
        "high_entropy_files": int(rng.integers(0, 3)),
        "entropy_average": float(rng.uniform(3.0, 6.5)),
        "suspicious_process": int(rng.choice([0, 0, 0, 1])),
        "backup_activity": 0,
        "network_connections": int(rng.integers(0, 6)),
        "label": 0,
    }

def build_dataset(n_per_class: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = [generate_sample(False, rng) for _ in range(n_per_class)]
    rows += [generate_sample(True, rng) for _ in range(n_per_class)]
    return pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)

def train_detector(dataset: pd.DataFrame, random_state: int = 42):
    missing = set(FEATURES + ["label"]) - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    clean = dataset[FEATURES + ["label"]].dropna().copy()
    X, y = clean[FEATURES], clean["label"].astype(int)
    if y.nunique() != 2:
        raise ValueError("Training data must contain both label 0 (normal) and 1 (ransomware-like).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=250, max_depth=12, class_weight="balanced",
        random_state=random_state, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    prob = rf.predict_proba(X_test)[:, 1]
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=random_state)
    iso.fit(X_train[y_train == 0])
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
    }
    return rf, iso, metrics, (X_test, y_test, pred)

def rule_score(features: dict):
    score, reasons = 0, []
    checks = [
        (features["files_modified"] > 100, 20, "Rapid mass file modification"),
        (features["files_renamed"] > 50, 20, "Mass file renaming"),
        (features["entropy_average"] > 7.0, 20, "High average file entropy"),
        (features["suspicious_process"] == 1, 15, "Suspicious process indicator"),
        (features["backup_activity"] == 1, 15, "Backup activity indicator"),
        (features["network_connections"] > 10, 10, "Unusual network connection count"),
        (features["ransom_extensions"] > 0, 10, "Ransomware-like file extension indicator"),
    ]
    for matched, points, reason in checks:
        if matched:
            score += points
            reasons.append(reason)
    return min(score, 100), reasons

def analyze_behavior(feature_dict: dict, rf_model, isolation_model):
    row = pd.DataFrame([{k: feature_dict[k] for k in FEATURES}], columns=FEATURES)
    ml_score = float(rf_model.predict_proba(row)[0, 1] * 100)
    anomaly = int(isolation_model.predict(row)[0] == -1)
    anomaly_score = 20 if anomaly else 0
    rules, reasons = rule_score(feature_dict)
    final = round(min(100.0, rules * 0.50 + ml_score * 0.40 + anomaly_score * 0.10), 2)
    severity = "NORMAL" if final <= 30 else "SUSPICIOUS" if final <= 60 else "HIGH RISK" if final <= 80 else "CRITICAL"
    if anomaly:
        reasons.append("Behavior differs from the normal training profile")
    return {
        "rule_score": rules, "ml_score": round(ml_score, 2),
        "anomaly_score": anomaly_score, "final_risk_score": final,
        "severity": severity, "reasons": reasons or ["No configured warning indicators triggered"],
        "alert": "CRITICAL: investigate immediately" if final > 80 else
                 "HIGH RISK: investigate" if final > 60 else
                 "SUSPICIOUS: monitor" if final > 30 else "NORMAL: no configured alert"
    }
