---
title: Industrial Predictive Maintenance - LightGBM + SHAP
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - predictive-maintenance
  - lightgbm
  - shap
  - explainable-ai
  - industry-4.0
  - machine-learning
---

# 🏭 Industrial Predictive Maintenance
### Explainable Ensemble Learning with LightGBM & SHAP

> **Authors:** Vibhor Kumar · Rahul Shankar Panday · Ridesh Rajesh  
> **Institution:** Department of Computer Science & Engineering, RVITM Bengaluru

---

## 📖 Overview

This application demonstrates a production-ready **predictive maintenance system** for Industry 4.0 environments. It uses a **LightGBM** gradient boosting model trained on the AI4I 2020 dataset to predict equipment failures in real time, with **SHAP** (SHapley Additive exPlanations) providing transparent, interpretable explanations for every prediction.

Unplanned downtime costs manufacturers ~$50 billion annually. Our system helps maintenance teams **act before failures happen**.

---

## 🎯 Model Performance

| Metric     | LightGBM ⭐ | Random Forest (Baseline) |
|------------|:-----------:|:------------------------:|
| Accuracy   | **97.85%**  | 97.25%                   |
| F1-Score   | **0.7514**  | 0.7027                   |
| Recall     | **95.59%**  | 95.59%                   |
| Precision  | **0.619**   | 0.556                    |

**5.71% F1-Score improvement** over baseline · **65/68 failures detected** in test set

---

## 🔑 Key Findings (SHAP Analysis)

The top three failure predictors identified by SHAP:

| Rank | Feature | SHAP Value | Insight |
|------|---------|------------|---------|
| 1 | **Torque [Nm]** | 1.5035 | Excessive torque = mechanical stress |
| 2 | **Tool Wear [min]** | 0.8527 | High wear correlates with failure |
| 3 | **Heat Dissipation Failure** | 0.7035 | Thermal issues are critical |

---

## 🚀 Features

| Tab | Description |
|-----|-------------|
| 🔍 **Single Prediction** | Enter 6 machine parameters → instant failure prediction with risk level + SHAP waterfall chart |
| 📂 **Batch Prediction** | Upload a CSV → bulk predictions + downloadable results |
| 📊 **Model Information** | Performance charts, confusion matrices, SHAP importance plot |
| ℹ️ **About** | Project description, team info, SDG 9 alignment |

---

## 🖱️ How to Use

### Single Prediction
1. Open the **🔍 Single Prediction** tab
2. Adjust the 6 sliders for your machine parameters
3. Click **🔮 Predict**
4. View the result, risk level, probability bar, and SHAP feature contribution chart
5. Try the **✅ Normal Operation** or **⚠️ High-Risk Scenario** example buttons

### Batch Prediction
1. Prepare a CSV with these columns:
   ```
   Type, Air_temp, Process_temp, Rotation_speed, Torque, Tool_wear
   ```
2. Open the **📂 Batch Prediction** tab and upload your file
3. Click **⚡ Run Batch Prediction**
4. View the summary and download results as CSV

---

## 📊 Input Feature Reference

| Feature | Range | Unit | Importance |
|---------|-------|------|------------|
| Type | 0, 1, 2 | — | L=0, M=1, H=2 |
| Air Temperature | 295–305 | K | Environmental |
| Process Temperature | 305–315 | K | Operational |
| Rotational Speed | 1200–2900 | rpm | Critical |
| **Torque** | 3–80 | Nm | ⭐ Most Important |
| **Tool Wear** | 0–250 | min | ⭐ 2nd Most Important |

---

## 🧪 Example Scenarios

**Normal Operation:**
```
Type=1, Air Temp=300 K, Proc Temp=310 K, Speed=1500 rpm, Torque=40 Nm, Wear=100 min
→ Expected: Normal (Low Risk)
```

**High-Risk Scenario:**
```
Type=0, Air Temp=302 K, Proc Temp=312 K, Speed=1200 rpm, Torque=75 Nm, Wear=220 min
→ Expected: Failure (High Risk)
```

---

## 🗂️ Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI ML Repository)

- 10,000 samples · 14 features · Binary classification
- Class imbalance: 96.61% Normal vs 3.39% Failure
- SMOTE applied to training data (1:1 ratio)
- Source: [UCI Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

---

## 🛠️ Technical Stack

```
LightGBM 4.5.0    – Gradient boosting classifier
SHAP 0.46.0       – Explainability (TreeExplainer)
scikit-learn 1.5  – Preprocessing, SMOTE
Gradio 4.44.0     – Web application framework
Pandas / NumPy    – Data processing
Matplotlib        – Visualisations
```

---

## 📁 Files

```
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── lgbm_model.pkl      # Trained LightGBM model
└── README.md           # This file
```

---

## 👥 Research Team

**Vibhor Kumar · Rahul Shankar Panday · Ridesh Rajesh**  
Department of Computer Science & Engineering  
RV Institute of Technology and Management, Bengaluru, Karnataka, India  

---

## 🌱 SDG 9 Alignment

This project advances **UN Sustainable Development Goal 9** (Industry, Innovation and Infrastructure) by enabling data-driven maintenance strategies that reduce resource waste, minimise environmental impact from equipment failures, and enhance operational efficiency.

---

## 📜 License

MIT License – Free for academic and research use.
