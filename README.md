# Ransomware Early Detection System

A defensive, simulation-based academic project that demonstrates ransomware-like behavior detection using machine learning, anomaly detection, and explainable rules.

## Safety and scope
This project does **not** contain ransomware, file-encryption routines, destructive simulation, persistence, process-killing, or host-isolation actions. The simulator generates synthetic behavior metrics only. The web dashboard analyzes manually entered or synthetic metrics; it does not monitor a real endpoint.

## Project structure
```text
Ransomware-Early-Detection/
├── Ransomware_Early_Detection.ipynb
├── app.py
├── ransomware_core.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
└── reports/
```

## Run in Google Colab
1. Upload `Ransomware_Early_Detection.ipynb` to Google Colab.
2. Select Runtime → Run all.
3. The notebook installs its dependencies and runs using synthetic data only.
4. No ngrok token or tunnel is needed for notebook execution.

## Run the web app locally (Windows)
```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit, usually http://localhost:8501.

## Run the model training script
```bash
python train_model.py
```

## Methodology
- Synthetic behavior dataset with normal (0) and ransomware-like (1) labels.
- Random Forest supervised classification.
- Isolation Forest anomaly detection trained on normal samples.
- Rule-based indicators.
- Hybrid risk score: 50% rule score + 40% Random Forest probability + 10% anomaly score.

## Limitations
All training and evaluation data are synthetic and deliberately separable for demonstration. Metrics are not evidence of real-world detection performance. For research-grade evaluation, replace synthetic data with a properly licensed, representative dataset and use a time- or host-separated test set.
