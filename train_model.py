from pathlib import Path
import joblib
from ransomware_core import build_dataset, train_detector

ROOT = Path(__file__).resolve().parent
dataset = build_dataset(n_per_class=1200, random_state=42)
rf, isolation, metrics, _ = train_detector(dataset)
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump({"random_forest": rf, "isolation_forest": isolation}, ROOT / "models" / "ransomware_models.joblib")
dataset.to_csv(ROOT / "synthetic_ransomware_behavior.csv", index=False)
print("Models saved to models/ransomware_models.joblib")
print("Evaluation metrics:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
print("Note: evaluation uses synthetic data only.")
