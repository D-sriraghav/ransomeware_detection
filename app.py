import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
from ransomware_core import FEATURES, build_dataset, train_detector, analyze_behavior

st.set_page_config(page_title="Ransomware Early Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Ransomware Early Detection System")
st.caption("Defensive, simulation-based behavioral analytics dashboard")

st.warning(
    "Educational prototype only. It analyzes manually entered or synthetic behavior metrics. "
    "It does not monitor endpoints, encrypt/decrypt files, isolate hosts, or stop processes."
)

@st.cache_resource
def get_detector():
    dataset = build_dataset(n_per_class=1200, random_state=42)
    rf, iso, metrics, split = train_detector(dataset)
    return dataset, rf, iso, metrics

dataset, rf_model, isolation_model, metrics = get_detector()

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a page", ["Dashboard", "Analyze behavior", "Dataset explorer", "Model evaluation", "About"])

if page == "Dashboard":
    st.subheader("System overview")
    a, b, c, d = st.columns(4)
    a.metric("Training examples", f"{len(dataset):,}")
    b.metric("Model", "Random Forest")
    c.metric("Hold-out accuracy", f"{metrics['accuracy']:.1%}")
    d.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.markdown("### Detection pipeline")
    st.write("Synthetic / supplied metrics → Random Forest probability + Isolation Forest anomaly flag + explainable rules → combined risk score.")
    st.info("The sample data is synthetic. Metrics measure performance on a held-out split of generated examples, not real-world ransomware.")

elif page == "Analyze behavior":
    st.subheader("Behavior analysis")
    st.write("Enter observed counts or load the prefilled simulation. Use only authorized, non-sensitive telemetry.")
    preset = st.selectbox("Example profile", ["Custom", "Normal activity example", "Ransomware-like simulation"])
    defaults = {
        "Normal activity example": [12, 2, 0, 2, 3, 0, 5, 0, 5.1, 0, 0, 2],
        "Ransomware-like simulation": [240, 180, 20, 15, 7, 160, 180, 210, 7.7, 1, 1, 24],
    }
    vals = defaults.get(preset, [10, 1, 0, 2, 2, 0, 4, 0, 5.0, 0, 0, 2])
    with st.form("behavior_form"):
        c1, c2, c3 = st.columns(3)
        files_modified = c1.number_input("Files modified", min_value=0, max_value=100000, value=int(vals[0]))
        files_renamed = c2.number_input("Files renamed", min_value=0, max_value=100000, value=int(vals[1]))
        files_deleted = c3.number_input("Files deleted", min_value=0, max_value=100000, value=int(vals[2]))
        directories = c1.number_input("Directories touched", min_value=0, max_value=10000, value=int(vals[3]))
        extensions = c2.number_input("Extensions affected", min_value=0, max_value=1000, value=int(vals[4]))
        ransom_ext = c3.number_input("Ransom-like extensions", min_value=0, max_value=100000, value=int(vals[5]))
        eps = c1.number_input("Events per second", min_value=0, max_value=100000, value=int(vals[6]))
        high_entropy = c2.number_input("High-entropy files", min_value=0, max_value=100000, value=int(vals[7]))
        entropy = c3.slider("Average entropy (0–8)", 0.0, 8.0, float(vals[8]), 0.1)
        suspicious = c1.selectbox("Suspicious process indicator", [0, 1], index=int(vals[9]))
        backup = c2.selectbox("Backup activity indicator", [0, 1], index=int(vals[10]))
        network = c3.number_input("Network connections", min_value=0, max_value=100000, value=int(vals[11]))
        submitted = st.form_submit_button("Analyze risk", type="primary")
    if submitted:
        features = {
            "files_modified": files_modified, "files_renamed": files_renamed,
            "files_deleted": files_deleted, "directories_touched": directories,
            "extensions_affected": extensions, "ransom_extensions": ransom_ext,
            "events_per_second": eps, "high_entropy_files": high_entropy,
            "entropy_average": entropy, "suspicious_process": suspicious,
            "backup_activity": backup, "network_connections": network,
        }
        result = analyze_behavior(features, rf_model, isolation_model)
        st.markdown("---")
        x, y = st.columns([1, 2])
        x.metric("Combined risk score", f"{result['final_risk_score']}/100")
        x.metric("Severity", result["severity"])
        y.progress(int(result["final_risk_score"]))
        y.subheader(result["alert"])
        st.write("**Rule score:**", result["rule_score"], "| **ML score:**", result["ml_score"], "| **Anomaly score:**", result["anomaly_score"])
        st.markdown("**Detection reasons**")
        for reason in result["reasons"]:
            st.write("•", reason)
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "features": features, "result": result}
        st.download_button("Download analysis report (JSON)", json.dumps(report, indent=2), "ransomware_analysis_report.json", "application/json")

elif page == "Dataset explorer":
    st.subheader("Synthetic dataset")
    st.caption("Generated demonstration data; not collected from real endpoints.")
    st.dataframe(dataset.head(100), use_container_width=True)
    st.bar_chart(dataset["label"].value_counts().rename(index={0:"Normal", 1:"Ransomware-like"}))
    st.download_button("Download synthetic dataset CSV", dataset.to_csv(index=False), "synthetic_ransomware_behavior.csv", "text/csv")

elif page == "Model evaluation":
    st.subheader("Held-out model evaluation")
    cols = st.columns(5)
    for col, (name, key) in zip(cols, [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1","f1"),("ROC-AUC","roc_auc")]):
        col.metric(name, f"{metrics[key]:.3f}")
    st.text(metrics["classification_report"])
    st.write("Confusion matrix (rows = actual, columns = predicted; labels 0=normal, 1=ransomware-like)")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["Actual normal","Actual ransomware-like"], columns=["Predicted normal","Predicted ransomware-like"]))

else:
    st.subheader("About this project")
    st.markdown("""
    **Objective:** demonstrate a defensive approach to detecting ransomware-like behavior using activity metrics.

    **Methods**
    - Random Forest supervised classifier
    - Isolation Forest anomaly detection trained on normal synthetic examples
    - Rule-based indicators for rapid modifications, renames, entropy, suspicious process flags, backup activity, network activity, and ransom-like extensions
    - Hybrid risk score combining rule, ML, and anomaly scores

    **Limitations:** the dataset is synthetic and its patterns are deliberately simplified. This is a prototype for academic demonstration, not production endpoint detection. Validate on a licensed, representative dataset before making operational claims.
    """)
