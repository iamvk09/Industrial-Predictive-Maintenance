"""
=============================================================================
Explainable Ensemble Learning for Industrial Predictive Maintenance
Using SHAP and LightGBM

Authors: Vibhor Kumar, Rahul Shankar Panday, Ridesh Rajesh
Institution: Department of Computer Science & Engineering, RVITM Bengaluru
=============================================================================
"""

import gradio as gr
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "Type",
    "Air_temp",
    "Process_temp",
    "Rotation_speed",
    "Torque",
    "Tool_wear",
]

FEATURE_RANGES = {
    "Type":            (0,    2),
    "Air_temp":        (295,  305),
    "Process_temp":    (305,  315),
    "Rotation_speed":  (1200, 2900),
    "Torque":          (3,    80),
    "Tool_wear":       (0,    250),
}

# SHAP importance values from paper (used for fallback visualisation if model is absent)
SHAP_IMPORTANCE = {
    "Torque":          1.5035,
    "Tool_wear":       0.8527,
    "Air_temp":        0.1523,
    "Process_temp":    0.3128,
    "Rotation_speed":  0.6745,
    "Type":            0.0892,
}

MODEL = None
EXPLAINER = None

def load_model():
    """Load the LightGBM model from disk; return True on success."""
    global MODEL, EXPLAINER
    try:
        with open("lgbm_model.pkl", "rb") as f:
            MODEL = pickle.load(f)
        # Build SHAP TreeExplainer for the loaded model
        EXPLAINER = shap.TreeExplainer(MODEL)
        print("[INFO] Model loaded successfully.")
        return True
    except FileNotFoundError:
        print("[WARNING] lgbm_model.pkl not found - running in demo mode.")
        return False
    except Exception as e:
        print(f"[WARNING] Model loading error: {e} - running in demo mode.")
        return False

MODEL_LOADED = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_level(prob: float) -> tuple[str, str, str]:
    """Return (label, text_color, background_color) for a given failure probability."""
    if prob >= 0.7:
        return "HIGH RISK", "#fca5a5", "rgba(220, 38, 38, 0.25)"
    elif prob >= 0.4:
        return "MEDIUM RISK", "#fde047", "rgba(217, 119, 6, 0.25)"
    else:
        return "LOW RISK", "#86efac", "rgba(22, 163, 74, 0.25)"


def predict_single(type_val, air_temp, proc_temp, rot_speed, torque, tool_wear):
    """
    Run inference for one sample.
    Returns: (result_html, shap_figure)
    """
    # ── Input validation ──────────────────────────────────────────────────
    errors = []
    inputs = {
        "Type":           type_val,
        "Air_temp":       air_temp,
        "Process_temp":   proc_temp,
        "Rotation_speed": rot_speed,
        "Torque":         torque,
        "Tool_wear":      tool_wear,
    }
    for feat, val in inputs.items():
        lo, hi = FEATURE_RANGES[feat]
        if not (lo <= val <= hi):
            errors.append(f"<li><b style='color:#f87171;'>{feat}</b>: {val} is outside valid range [{lo}, {hi}]</li>")

    if errors:
        html = f"""
        <div class='ipm-card' style='background:rgba(239, 68, 68, 0.12);border:1px solid #ef4444;border-radius:12px;padding:20px;'>
          <h3 style='color:#f87171;margin:0 0 10px 0;font-size:1.2rem;font-weight:700;'>⚠️ Input Validation Error</h3>
          <ul style='margin:0;padding-left:20px;color:#fca5a5;line-height:1.6;'>{''.join(errors)}</ul>
        </div>"""
        return html, None

    # ── Build feature array ───────────────────────────────────────────────
    X = np.array([[type_val, air_temp, proc_temp, rot_speed, torque, tool_wear]])
    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    # ── Inference ─────────────────────────────────────────────────────────
    if MODEL_LOADED and MODEL is not None:
        prob = float(MODEL.predict_proba(df)[0][1])
        prediction = int(MODEL.predict(df)[0])
    else:
        # Demo mode heuristic
        score = 0.0
        score += max(0, (torque - 50) / 30) * 0.5
        score += max(0, (tool_wear - 150) / 100) * 0.3
        score += max(0, (2900 - rot_speed) / 1700) * 0.1
        score += (1 - type_val / 2) * 0.1
        prob = min(float(score), 0.99)
        prediction = 1 if prob >= 0.5 else 0

    risk_label, risk_text_color, risk_bg_color = get_risk_level(prob)
    pred_text = "FAILURE PREDICTED" if prediction == 1 else "NORMAL OPERATION"
    pred_color = "#f87171" if prediction == 1 else "#4ade80"
    pred_bg = "rgba(239, 68, 68, 0.18)" if prediction == 1 else "rgba(34, 197, 94, 0.18)"
    pred_border = "#ef4444" if prediction == 1 else "#22c55e"
    pred_icon = "⚠️" if prediction == 1 else "✅"
    pct = prob * 100

    # ── Result HTML with High Contrast Dark Theme ─────────────────────────
    html = f"""
    <div style='font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;'>
      <!-- Main prediction banner -->
      <div style='background:{pred_bg};border:2px solid {pred_border};
                  border-radius:12px;padding:22px;margin-bottom:16px;text-align:center;'>
        <div style='font-size:1.8rem;margin-bottom:6px;'>{pred_icon}</div>
        <h2 style='color:{pred_color};margin:0 0 8px 0;font-size:1.6rem;font-weight:800;letter-spacing:0.5px;'>
          {pred_text}
        </h2>
        <p style='margin:0;font-size:1.15rem;color:#f1f5f9;font-weight:500;'>
          Failure Probability: <b style='color:{pred_color};font-size:1.35rem;'>{pct:.1f}%</b>
        </p>
      </div>

      <!-- Risk level assessment -->
      <div style='background:{risk_bg_color};border:1px solid {risk_text_color};
                  border-radius:10px;padding:12px 16px;margin-bottom:16px;
                  display:flex;align-items:center;justify-content:center;gap:10px;'>
        <span style='color:#94a3b8;font-size:0.95rem;font-weight:600;'>RISK ASSESSMENT:</span>
        <span style='color:{risk_text_color};font-size:1.2rem;font-weight:800;letter-spacing:0.8px;'>
          {risk_label}
        </span>
      </div>

      <!-- Probability bar -->
      <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px;margin-bottom:16px;'>
        <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
          <span style='color:#f8fafc;font-weight:600;font-size:0.95rem;'>Failure Likelihood Indicator</span>
          <span style='color:{pred_color};font-weight:700;font-size:0.95rem;'>{pct:.1f}%</span>
        </div>
        <div style='background:#0f172a;border-radius:999px;height:18px;overflow:hidden;border:1px solid #334155;'>
          <div style='width:{pct:.1f}%;background:linear-gradient(90deg, #22c55e, #eab308 60%, #ef4444);
                      height:100%;border-radius:999px;transition:width 0.4s ease;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;color:#94a3b8;font-size:0.82rem;margin-top:6px;'>
          <span>0% (Safe / Low Risk)</span>
          <span>50%</span>
          <span>100% (Critical)</span>
        </div>
      </div>

      <!-- Risk thresholds reference -->
      <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px;'>
        <p style='margin:0 0 10px 0;font-weight:700;color:#f8fafc;font-size:0.92rem;text-transform:uppercase;letter-spacing:0.5px;'>
          Operational Thresholds
        </p>
        <div style='display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:10px;font-size:0.88rem;'>
          <div style='background:#0f172a;padding:8px 10px;border-radius:6px;border-left:3px solid #22c55e;'>
            <span style='color:#86efac;font-weight:700;'>LOW RISK</span><br>
            <span style='color:#cbd5e1;'>&lt; 40%</span>
          </div>
          <div style='background:#0f172a;padding:8px 10px;border-radius:6px;border-left:3px solid #eab308;'>
            <span style='color:#fde047;font-weight:700;'>MEDIUM RISK</span><br>
            <span style='color:#cbd5e1;'>40% – 70%</span>
          </div>
          <div style='background:#0f172a;padding:8px 10px;border-radius:6px;border-left:3px solid #ef4444;'>
            <span style='color:#fca5a5;font-weight:700;'>HIGH RISK</span><br>
            <span style='color:#cbd5e1;'>&gt; 70%</span>
          </div>
        </div>
      </div>

      {"<p style='margin-top:12px;color:#f59e0b;font-size:0.85rem;background:rgba(245,158,11,0.1);padding:8px 12px;border-radius:6px;'>⚠️ Running in demo fallback mode (lgbm_model.pkl not found). Real predictions require the trained model.</p>" if not MODEL_LOADED else ""}
    </div>
    """

    # ── SHAP explanation plot ─────────────────────────────────────────────
    fig = _shap_waterfall_plot(df, prob)
    return html, fig


def _shap_waterfall_plot(df: pd.DataFrame, prob: float) -> plt.Figure:
    """Generate a horizontal SHAP waterfall / contribution chart styled for dark mode."""
    if MODEL_LOADED and EXPLAINER is not None:
        shap_vals = EXPLAINER.shap_values(df)
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        else:
            vals = shap_vals[0]
    else:
        row = df.iloc[0]
        vals = []
        for feat in FEATURE_NAMES:
            lo, hi = FEATURE_RANGES[feat]
            mid = (lo + hi) / 2
            scale = (row[feat] - mid) / max(hi - mid, 1)
            vals.append(SHAP_IMPORTANCE[feat] * scale)
        vals = np.array(vals)

    # Sort by absolute value descending
    order = np.argsort(np.abs(vals))[::-1]
    sorted_feats = [FEATURE_NAMES[i] for i in order]
    sorted_vals = vals[order]
    colours = ["#ef4444" if v > 0 else "#38bdf8" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#0f172a")
    bars = ax.barh(sorted_feats[::-1], sorted_vals[::-1], color=colours[::-1],
                   edgecolor="#334155", height=0.62)

    # Value labels with high contrast
    for bar, val in zip(bars, sorted_vals[::-1]):
        offset = 0.02 if val >= 0 else -0.02
        xpos = bar.get_width() + offset
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=9.5, fontweight="600", color="#f8fafc")

    ax.axvline(0, color="#64748b", linewidth=1.2, linestyle="--")
    ax.set_xlabel("SHAP Impact on Failure Log-Odds", fontsize=9.5, color="#cbd5e1", fontweight="500")
    ax.set_title(f"Feature Attribution | Predicted Failure Risk: {prob*100:.1f}%",
                 fontsize=11.5, fontweight="bold", color="#38bdf8", pad=12)
    ax.tick_params(axis="y", labelsize=9.5, colors="#f1f5f9")
    ax.tick_params(axis="x", labelsize=8.5, colors="#cbd5e1")

    # Spines styling
    for spine in ax.spines.values():
        spine.set_color("#334155")

    red_patch = mpatches.Patch(color="#ef4444", label="↑ Increases failure risk")
    blue_patch = mpatches.Patch(color="#38bdf8", label="↓ Decreases failure risk")
    legend = ax.legend(handles=[red_patch, blue_patch], fontsize=8.5, loc="lower right",
                       facecolor="#1e293b", edgecolor="#475569")
    for text in legend.get_texts():
        text.set_color("#f8fafc")

    ax.set_facecolor("#1e293b")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(file_obj):
    """Accept a CSV upload, return (summary_html, result_dataframe, csv_bytes)."""
    if file_obj is None:
        return """
        <div style='background:rgba(239,68,68,0.12);border:1px solid #ef4444;border-radius:10px;padding:16px;'>
          <b style='color:#f87171;'>⚠️ Please select or upload a CSV file before running batch prediction.</b>
        </div>
        """, None, None

    try:
        df_raw = pd.read_csv(file_obj.name)
    except Exception as e:
        return f"""
        <div style='background:rgba(239,68,68,0.12);border:1px solid #ef4444;border-radius:10px;padding:16px;'>
          <b style='color:#f87171;'>⚠️ Error reading CSV file:</b>
          <p style='color:#fca5a5;margin:4px 0 0 0;'>{str(e)}</p>
        </div>
        """, None, None

    # Direct column matching against model's expected names
    col_map = {}
    for need in FEATURE_NAMES:
        for col in df_raw.columns:
            if col == need or col.strip().lower() == need.lower():
                col_map[need] = col
                break

    missing = [f for f in FEATURE_NAMES if f not in col_map]
    if missing:
        hint = ", ".join(FEATURE_NAMES)
        return (
            f"""
            <div style='background:rgba(239,68,68,0.12);border:1px solid #ef4444;border-radius:10px;padding:16px;'>
              <h4 style='color:#f87171;margin:0 0 8px 0;'>⚠️ Missing Required Columns</h4>
              <p style='color:#fca5a5;margin:0 0 6px 0;'>Missing columns in uploaded file: <b style='color:#ffffff;'>{', '.join(missing)}</b></p>
              <p style='color:#cbd5e1;margin:0;font-size:0.9rem;'>Required columns are: <code style='background:#1e293b;color:#38bdf8;padding:2px 6px;border-radius:4px;'>{hint}</code></p>
            </div>
            """,
            None, None,
        )

    df = df_raw[[col_map[f] for f in FEATURE_NAMES]].copy()
    df.columns = FEATURE_NAMES

    # Inference
    if MODEL_LOADED and MODEL is not None:
        probs = MODEL.predict_proba(df)[:, 1]
        preds = MODEL.predict(df)
    else:
        probs, preds = [], []
        for _, row in df.iterrows():
            score = 0.0
            score += max(0, (row["Torque"] - 50) / 30) * 0.5
            score += max(0, (row["Tool_wear"] - 150) / 100) * 0.3
            score += max(0, (2900 - row["Rotation_speed"]) / 1700) * 0.1
            score += (1 - row["Type"] / 2) * 0.1
            p = min(float(score), 0.99)
            probs.append(p)
            preds.append(1 if p >= 0.5 else 0)
        probs = np.array(probs)
        preds = np.array(preds)

    result_df = df.copy()
    result_df["Failure_Probability"] = (probs * 100).round(2)
    result_df["Prediction"]          = ["Failure" if p == 1 else "Normal" for p in preds]
    result_df["Risk_Level"]          = [get_risk_level(p)[0] for p in probs]

    total      = len(result_df)
    n_fail     = int(preds.sum())
    n_normal   = total - n_fail
    fail_rate  = (n_fail / total * 100) if total > 0 else 0
    avg_prob   = (probs.mean() * 100) if total > 0 else 0
    high_risk  = int((probs >= 0.7).sum())
    med_risk   = int(((probs >= 0.4) & (probs < 0.7)).sum())
    low_risk   = int((probs < 0.4).sum())

    rate_color = "#f87171" if fail_rate > 10 else "#fde047" if fail_rate > 5 else "#4ade80"

    summary_html = f"""
    <div style='font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;'>
      <div style='background:#1e293b;border:2px solid #3b82f6;border-radius:12px;padding:20px;margin-bottom:16px;'>
        <h3 style='color:#38bdf8;margin:0 0 16px 0;font-size:1.25rem;font-weight:700;'>
          📊 Batch Prediction Summary
        </h3>
        <div style='display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:12px;'>
          <div style='background:#0f172a;border:1px solid #334155;border-radius:8px;padding:14px;text-align:center;'>
            <div style='font-size:1.9rem;font-weight:800;color:#f8fafc;'>{total:,}</div>
            <div style='color:#94a3b8;font-size:0.85rem;font-weight:600;margin-top:2px;'>Total Processed Samples</div>
          </div>
          <div style='background:#0f172a;border:1px solid #334155;border-radius:8px;padding:14px;text-align:center;'>
            <div style='font-size:1.9rem;font-weight:800;color:#f87171;'>{n_fail:,}</div>
            <div style='color:#94a3b8;font-size:0.85rem;font-weight:600;margin-top:2px;'>Failures Predicted</div>
          </div>
          <div style='background:#0f172a;border:1px solid #334155;border-radius:8px;padding:14px;text-align:center;'>
            <div style='font-size:1.9rem;font-weight:800;color:#4ade80;'>{n_normal:,}</div>
            <div style='color:#94a3b8;font-size:0.85rem;font-weight:600;margin-top:2px;'>Normal Operations</div>
          </div>
          <div style='background:#0f172a;border:1px solid #334155;border-radius:8px;padding:14px;text-align:center;'>
            <div style='font-size:1.9rem;font-weight:800;color:{rate_color};'>{fail_rate:.1f}%</div>
            <div style='color:#94a3b8;font-size:0.85rem;font-weight:600;margin-top:2px;'>Failure Rate</div>
          </div>
          <div style='background:#0f172a;border:1px solid #334155;border-radius:8px;padding:14px;text-align:center;'>
            <div style='font-size:1.9rem;font-weight:800;color:#c084fc;'>{avg_prob:.1f}%</div>
            <div style='color:#94a3b8;font-size:0.85rem;font-weight:600;margin-top:2px;'>Average Failure Probability</div>
          </div>
          <div style='background:#0f172a;border:1px solid #334155;border-radius:8px;padding:14px;text-align:center;'>
            <div style='font-size:1.05rem;font-weight:700;color:#f8fafc;padding-top:6px;'>
              🔴 {high_risk} &nbsp; 🟡 {med_risk} &nbsp; 🟢 {low_risk}
            </div>
            <div style='color:#94a3b8;font-size:0.85rem;font-weight:600;margin-top:6px;'>Risk Distribution</div>
          </div>
        </div>
      </div>
      {"<p style='color:#f59e0b;font-size:0.85rem;background:rgba(245,158,11,0.1);padding:8px 12px;border-radius:6px;'>⚠️ Demo mode active - upload lgbm_model.pkl for real LightGBM inferences.</p>" if not MODEL_LOADED else ""}
    </div>
    """

    return summary_html, result_df, result_df.to_csv(index=False)


def download_results(csv_bytes):
    """Return path to a temporary CSV file for download."""
    if not csv_bytes:
        return None
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, "predictive_maintenance_results.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(csv_bytes)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL INFO FIGURES (THEMED FOR DARK / CONTRAST)
# ─────────────────────────────────────────────────────────────────────────────

def _model_performance_fig() -> plt.Figure:
    """Bar chart of model comparison metrics from the paper."""
    models = ["RF\n(Baseline)", "SVM\n(Baseline)", "XGBoost", "LightGBM\n★ Proposed", "Stacking"]
    accuracy = [97.25, 79.10, 97.80, 97.85, 97.75]
    f1_score  = [0.703,  0.226,  0.747,  0.751,  0.743]

    x = np.arange(len(models))
    w = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 4.5), facecolor="#0f172a")

    bars1 = ax1.bar(x - w/2, accuracy, w, label="Accuracy (%)",
                    color=["#38bdf8" if i != 3 else "#60a5fa" for i in range(len(models))],
                    edgecolor="#1e293b")
    ax1.set_ylabel("Accuracy (%)", color="#38bdf8", fontsize=10, fontweight="600")
    ax1.set_ylim(70, 102)
    ax1.tick_params(axis="y", labelcolor="#38bdf8")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, f1_score, w, label="F1-Score",
                    color=["#f87171" if i != 3 else "#ef4444" for i in range(len(models))],
                    edgecolor="#1e293b")
    ax2.set_ylabel("F1-Score", color="#f87171", fontsize=10, fontweight="600")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#f87171")

    # Value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold", color="#f8fafc")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold", color="#fca5a5")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9.5, fontweight="600", color="#f1f5f9")
    ax1.set_title("Model Performance Comparison (AI4I Test Set)", fontsize=12, fontweight="bold",
                  color="#38bdf8", pad=12)

    for spine in ax1.spines.values():
        spine.set_color("#334155")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    lines = [
        mpatches.Patch(color="#38bdf8", label="Accuracy (%)"),
        mpatches.Patch(color="#ef4444", label="F1-Score"),
    ]
    legend = ax1.legend(handles=lines, loc="lower right", fontsize=9,
                        facecolor="#1e293b", edgecolor="#475569")
    for text in legend.get_texts():
        text.set_color("#f8fafc")

    ax1.set_facecolor("#1e293b")
    plt.tight_layout()
    return fig


def _shap_importance_fig() -> plt.Figure:
    """Horizontal bar chart of SHAP feature importances from the research paper."""
    feats = [
        "Type", "Air Temperature [K]", "TWF Flag",
        "Process Temp [K]", "OSF Flag", "Rotational Speed",
        "PWF Flag", "HDF Flag", "Tool Wear [min]", "Torque [Nm]",
    ]
    vals = [0.0892, 0.1523, 0.2845, 0.3128, 0.4521, 0.6745, 0.6892, 0.7035, 0.8527, 1.5035]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
    colours = ["#38bdf8" if v < 0.7 else "#60a5fa" if v < 1.0 else "#ef4444" for v in vals]
    bars = ax.barh(feats, vals, color=colours, edgecolor="#1e293b", height=0.65)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9, fontweight="bold", color="#f8fafc")

    ax.set_xlabel("Mean |SHAP Value|", fontsize=10, color="#cbd5e1", fontweight="600")
    ax.set_title("Global Feature Importance (SHAP Values)", fontsize=12,
                 fontweight="bold", color="#38bdf8", pad=10)
    ax.set_xlim(0, 1.8)
    ax.tick_params(axis="y", labelsize=9.5, colors="#f1f5f9")
    ax.tick_params(axis="x", labelsize=8.5, colors="#cbd5e1")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Highlight top-3
    for bar, feat in zip(bars, feats):
        if feat in ("Torque [Nm]", "Tool Wear [min]", "HDF Flag"):
            ax.text(0.02, bar.get_y() + bar.get_height()/2,
                    "★", color="#f59e0b", fontsize=11, va="center")

    ax.set_facecolor("#1e293b")
    plt.tight_layout()
    return fig


def _confusion_matrix_fig() -> plt.Figure:
    """Side-by-side confusion matrices for RF vs LightGBM."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), facecolor="#0f172a")

    data = [
        ("Random Forest (Baseline)", np.array([[1891, 41], [3, 65]]),  "Greys"),
        ("LightGBM ★ (Proposed)",   np.array([[1905, 27], [3, 65]]),  "Blues"),
    ]

    for ax, (title, cm, cmap_name) in zip(axes, data):
        cmap = plt.get_cmap(cmap_name)
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color="#38bdf8", pad=8)
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Normal", "Failure"], fontsize=9, color="#f1f5f9")
        ax.set_yticklabels(["Normal", "Failure"], fontsize=9, color="#f1f5f9")
        ax.set_xlabel("Predicted Label", fontsize=8.5, color="#cbd5e1", labelpad=6)
        ax.set_ylabel("True Label", fontsize=8.5, color="#cbd5e1", labelpad=6)

        for spine in ax.spines.values():
            spine.set_color("#334155")

        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "#ffffff" if val > thresh else "#0f172a"
                ax.text(j, i, str(val),
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color=color)

    fig.suptitle("Test Set Confusion Matrix Comparison (N=2,000)", fontsize=11.5,
                 fontweight="bold", color="#f8fafc", y=1.02)
    plt.tight_layout()
    return fig


# Pre-render static figures
perf_fig = _model_performance_fig()
shap_fig = _shap_importance_fig()
cm_fig   = _confusion_matrix_fig()

# ─────────────────────────────────────────────────────────────────────────────
# 5. GRADIO UI THEME & CSS
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* Top level container & typography */
body, .gradio-container {
    background-color: #0b1120 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: #f1f5f9 !important;
}

/* Tab Header Strip */
.tab-nav {
    background: #0f172a !important;
    border-bottom: 2px solid #1e293b !important;
    padding: 6px 10px 0 10px !important;
    gap: 8px !important;
}

.tab-nav button {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-top-left-radius: 8px !important;
    border-top-right-radius: 8px !important;
    padding: 10px 18px !important;
    background: transparent !important;
    transition: all 0.2s ease-in-out !important;
}

.tab-nav button:hover {
    color: #ffffff !important;
    background: rgba(56, 189, 248, 0.1) !important;
}

.tab-nav button.selected {
    color: #38bdf8 !important;
    background: #1e293b !important;
    border-bottom: 3px solid #38bdf8 !important;
}

/* Headings in Dark Theme */
.dark h1, .dark h2, .dark h3, .dark h4,
h1, h2, h3, h4 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

.section-title {
    color: #38bdf8 !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    margin: 12px 0 8px 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

/* Form Controls & Slider labels */
label span, .gr-form label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

.gr-form .gr-form-sublabel, .info {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

/* Inputs and sliders backgrounds */
.gr-box, .gr-panel, div[data-testid="block"] {
    border-radius: 10px !important;
}

input[type="number"], input[type="text"], textarea {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
}

/* Buttons styling */
button.primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
}

button.secondary {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #475569 !important;
    font-weight: 600 !important;
}

button.secondary:hover {
    background: #334155 !important;
    color: #ffffff !important;
}

/* Dataframe Styling */
table.gr-dataframe {
    color: #f1f5f9 !important;
    background-color: #0f172a !important;
}

table.gr-dataframe th {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #334155 !important;
}

table.gr-dataframe td {
    border-bottom: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
}

/* Plot containers */
.gr-plot {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .tab-nav button {
        padding: 8px 12px !important;
        font-size: 0.85rem !important;
    }
}
"""

HEADER_HTML = """
<div style='background:linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
            color:white;padding:26px 30px;border-radius:14px;margin-bottom:12px;
            box-shadow:0 8px 24px rgba(15, 23, 42, 0.4);border:1px solid #1e40af;'>
  <div style='display:flex;align-items:center;gap:18px;'>
    <div style='font-size:3.2rem;'>🏭</div>
    <div>
      <h1 style='margin:0;font-size:1.85rem;font-weight:800;letter-spacing:-0.5px;color:#ffffff;'>
        Industrial Predictive Maintenance
      </h1>
      <p style='margin:6px 0 0 0;font-size:1.05rem;color:#bfdbfe;font-weight:500;'>
        Explainable Machine Learning with LightGBM &amp; SHAP Interpretability
      </p>
    </div>
  </div>
  <div style='margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;'>
    <span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);padding:4px 12px;border-radius:999px;font-size:0.85rem;font-weight:600;color:#ffffff;'>
      📊 Accuracy: 97.85%
    </span>
    <span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);padding:4px 12px;border-radius:999px;font-size:0.85rem;font-weight:600;color:#ffffff;'>
      🎯 F1-Score: 0.7514
    </span>
    <span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);padding:4px 12px;border-radius:999px;font-size:0.85rem;font-weight:600;color:#ffffff;'>
      🔍 Recall: 95.59%
    </span>
    <span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);padding:4px 12px;border-radius:999px;font-size:0.85rem;font-weight:600;color:#ffffff;'>
      ⚡ LightGBM + SHAP
    </span>
    <span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);padding:4px 12px;border-radius:999px;font-size:0.85rem;font-weight:600;color:#ffffff;'>
      🌱 UN SDG 9
    </span>
  </div>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
# 6. APP BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0b1120",
        body_background_fill_dark="#0b1120",
        block_background_fill="#0f172a",
        block_background_fill_dark="#0f172a",
        block_border_color="#1e293b",
        block_border_color_dark="#1e293b",
        block_label_text_color="#f8fafc",
        block_label_text_color_dark="#f8fafc",
        input_background_fill="#1e293b",
        input_background_fill_dark="#1e293b",
        input_border_color="#334155",
        input_border_color_dark="#334155",
    )

    with gr.Blocks(css=CUSTOM_CSS, theme=custom_theme, title="Industrial Predictive Maintenance - RVITM") as demo:

        gr.HTML(HEADER_HTML)

        # ── Status banner ─────────────────────────────────────────────────
        if MODEL_LOADED:
            status_banner_html = """
            <div style='background:rgba(16, 185, 129, 0.15);border:1px solid #10b981;
                        border-radius:8px;padding:12px 18px;margin-bottom:12px;display:flex;align-items:center;gap:10px;'>
              <span style='font-size:1.3rem;'>✅</span>
              <div>
                <b style='color:#34d399;font-size:0.95rem;'>Model Status:</b>
                <span style='color:#f8fafc;font-size:0.95rem;margin-left:6px;'>
                  Model loaded successfully &mdash; real predictions active.
                </span>
              </div>
            </div>
            """
        else:
            status_banner_html = """
            <div style='background:rgba(245, 158, 11, 0.15);border:1px solid #f59e0b;
                        border-radius:8px;padding:12px 18px;margin-bottom:12px;display:flex;align-items:center;gap:10px;'>
              <span style='font-size:1.3rem;'>⚠️</span>
              <div>
                <b style='color:#fbbf24;font-size:0.95rem;'>Model Status:</b>
                <span style='color:#f8fafc;font-size:0.95rem;margin-left:6px;'>
                  Demo mode active &mdash; <code>lgbm_model.pkl</code> not found. Upload model to enable live predictions.
                </span>
              </div>
            </div>
            """
        gr.HTML(status_banner_html)

        # ══════════════════════════════════════════════════════════════════
        # TAB 1 – Single Prediction
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🔍 Single Prediction"):
            gr.HTML("<p style='color:#cbd5e1;margin:0 0 16px 0;font-size:1rem;'>"
                    "Adjust sensor telemetry parameters below to perform real-time failure inference "
                    "with SHAP feature attribution.</p>")

            with gr.Row():
                # ── Left: inputs ─────────────────────────────────────────
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>⚙️ Sensor Telemetry Inputs</div>")

                    type_input = gr.Slider(
                        minimum=0, maximum=2, step=1, value=1,
                        label="Product Type  [0 = Low | 1 = Medium | 2 = High]",
                        info="Quality grading of the machine component",
                    )
                    air_temp_input = gr.Slider(
                        minimum=295, maximum=305, step=0.1, value=300,
                        label="Air Temperature [K]",
                        info="Ambient temperature measured in Kelvin (295–305 K)",
                    )
                    proc_temp_input = gr.Slider(
                        minimum=305, maximum=315, step=0.1, value=310,
                        label="Process Temperature [K]",
                        info="Active process temperature measured in Kelvin (305–315 K)",
                    )
                    rot_speed_input = gr.Slider(
                        minimum=1200, maximum=2900, step=10, value=1500,
                        label="Rotational Speed [rpm]",
                        info="Drive spindle rotational velocity (1200–2900 rpm)",
                    )
                    torque_input = gr.Slider(
                        minimum=3, maximum=80, step=0.5, value=40,
                        label="Torque [Nm]  ⭐ Primary Predictor",
                        info="Applied torque in Newton-meters (3–80 Nm)",
                    )
                    tool_wear_input = gr.Slider(
                        minimum=0, maximum=250, step=1, value=100,
                        label="Tool Wear [min]  ⭐ Secondary Predictor",
                        info="Cumulative tool usage duration before replacement (0–250 min)",
                    )

                    with gr.Row():
                        predict_btn = gr.Button("🔮 Predict Failure Risk", variant="primary", scale=2)
                        clear_btn   = gr.Button("🔄 Reset Values",  variant="secondary", scale=1)

                    # Quick-load scenario presets
                    gr.HTML("<div class='section-title' style='margin-top:18px;'>💡 Instant Benchmark Scenarios</div>")
                    with gr.Row():
                        ex_normal_btn  = gr.Button("✅ Normal Operation Preset",  size="sm")
                        ex_failure_btn = gr.Button("⚠️ High-Risk Preset", size="sm")

                # ── Right: outputs ────────────────────────────────────────
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>📋 Prediction Result</div>")
                    result_html = gr.HTML()
                    gr.HTML("<div class='section-title' style='margin-top:16px;'>🔬 SHAP Explanation</div>")
                    shap_plot   = gr.Plot(label="Feature Contributions")

            # ── Event Wiring ─────────────────────────────────────────────
            inputs  = [type_input, air_temp_input, proc_temp_input,
                       rot_speed_input, torque_input, tool_wear_input]
            outputs = [result_html, shap_plot]

            predict_btn.click(fn=predict_single, inputs=inputs, outputs=outputs)

            clear_btn.click(
                fn=lambda: (1, 300, 310, 1500, 40, 100),
                outputs=[type_input, air_temp_input, proc_temp_input,
                         rot_speed_input, torque_input, tool_wear_input],
            )

            # Normal preset: Type=1, Air=300, Proc=310, Speed=1500, Torque=40, Wear=100
            ex_normal_btn.click(
                fn=lambda: (1, 300.0, 310.0, 1500, 40.0, 100),
                outputs=[type_input, air_temp_input, proc_temp_input,
                         rot_speed_input, torque_input, tool_wear_input],
            )
            # High-risk preset: Type=0, Air=302, Proc=312, Speed=1200, Torque=75, Wear=220
            ex_failure_btn.click(
                fn=lambda: (0, 302.0, 312.0, 1200, 75.0, 220),
                outputs=[type_input, air_temp_input, proc_temp_input,
                         rot_speed_input, torque_input, tool_wear_input],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 2 – Batch Prediction
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📂 Batch Prediction"):
            gr.HTML("""
            <p style='color:#cbd5e1;margin:0 0 10px 0;font-size:1rem;'>
              Upload sensor log CSV files for batch automated inference, failure classification, and downloadable export.
            </p>
            <div style='background:#1e293b;border:1px solid #3b82f6;border-radius:10px;
                        padding:14px 18px;margin-bottom:18px;'>
              <b style='color:#38bdf8;font-size:0.95rem;'>📋 Required CSV Column Names:</b><br>
              <code style='color:#ffffff;background:#0f172a;padding:4px 8px;border-radius:6px;display:inline-block;margin-top:6px;'>
                Type, Air_temp, Process_temp, Rotation_speed, Torque, Tool_wear
              </code>
            </div>
            """)

            with gr.Row():
                file_input    = gr.File(label="📁 Select or Drag CSV File", file_types=[".csv"])
                batch_run_btn = gr.Button("⚡ Run Batch Prediction", variant="primary")

            batch_summary = gr.HTML()

            with gr.Row():
                result_table  = gr.Dataframe(label="Prediction Results Table", interactive=False)

            with gr.Row():
                csv_state     = gr.State()
                download_btn  = gr.Button("⬇️ Download Results CSV", variant="secondary")
                download_file = gr.File(label="Download Ready File", visible=False)

            batch_run_btn.click(
                fn=predict_batch,
                inputs=[file_input],
                outputs=[batch_summary, result_table, csv_state],
            )
            download_btn.click(
                fn=download_results,
                inputs=[csv_state],
                outputs=[download_file],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 – Model Info
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📊 Model Information"):
            gr.HTML("""
            <div style='background:#1e293b;border:1px solid #3b82f6;border-radius:12px;
                        padding:22px;margin-bottom:18px;'>
              <h3 style='color:#38bdf8;margin:0 0 14px 0;font-size:1.3rem;font-weight:700;'>
                🤖 LightGBM Architecture &amp; Performance Summary
              </h3>
              <div style='display:grid;grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));gap:16px;'>
                <div style='background:#0f172a;padding:16px;border-radius:8px;border:1px solid #334155;'>
                  <b style='color:#94a3b8;'>Model Family:</b>
                  <span style='color:#f8fafc;font-weight:600;'> LightGBM (Gradient Boosting)</span><br><br>
                  <b style='color:#94a3b8;'>Hyperparameters:</b>
                  <span style='color:#f8fafc;'> 100 estimators, max depth 6</span><br><br>
                  <b style='color:#94a3b8;'>Learning Rate:</b>
                  <span style='color:#f8fafc;'> 0.1</span><br><br>
                  <b style='color:#94a3b8;'>Class Imbalance Strategy:</b>
                  <span style='color:#f8fafc;'> SMOTE (1:1 synthetic balancing)</span>
                </div>
                <div style='background:#0f172a;padding:16px;border-radius:8px;border:1px solid #334155;'>
                  <b style='color:#94a3b8;'>Test Accuracy:</b>
                  <span style='color:#4ade80;font-weight:800;font-size:1.1rem;'> 97.85%</span><br><br>
                  <b style='color:#94a3b8;'>F1-Score:</b>
                  <span style='color:#4ade80;font-weight:800;font-size:1.1rem;'> 0.7514</span><br><br>
                  <b style='color:#94a3b8;'>Recall:</b>
                  <span style='color:#4ade80;font-weight:800;font-size:1.1rem;'> 95.59% (65/68 failures detected)</span><br><br>
                  <b style='color:#94a3b8;'>Precision:</b>
                  <span style='color:#4ade80;font-weight:800;font-size:1.1rem;'> 0.6190</span>
                </div>
              </div>
            </div>
            """)

            with gr.Row():
                gr.Plot(value=perf_fig, label="Model Performance Comparison")
                gr.Plot(value=shap_fig, label="Global SHAP Feature Importance")

            gr.Plot(value=cm_fig, label="Confusion Matrices: RF vs LightGBM")

            gr.HTML("""
            <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;
                        padding:20px;margin-top:18px;'>
              <h4 style='color:#38bdf8;margin:0 0 12px 0;font-size:1.15rem;font-weight:700;'>📁 Dataset Specifications</h4>
              <div style='overflow-x:auto;'>
                <table style='width:100%;border-collapse:collapse;font-size:0.92rem;color:#f1f5f9;'>
                  <thead>
                    <tr style='background:#0f172a;border-bottom:2px solid #334155;'>
                      <th style='padding:10px 14px;text-align:left;color:#38bdf8;'>Property</th>
                      <th style='padding:10px 14px;text-align:left;color:#38bdf8;'>Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style='border-bottom:1px solid #334155;'>
                      <td style='padding:10px 14px;font-weight:600;'>Benchmark Dataset</td>
                      <td style='padding:10px 14px;'>AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)</td>
                    </tr>
                    <tr style='background:#0f172a;border-bottom:1px solid #334155;'>
                      <td style='padding:10px 14px;font-weight:600;'>Total Recorded Samples</td>
                      <td style='padding:10px 14px;'>10,000 observations</td>
                    </tr>
                    <tr style='border-bottom:1px solid #334155;'>
                      <td style='padding:10px 14px;font-weight:600;'>Normal Operations</td>
                      <td style='padding:10px 14px;'>9,661 instances (96.61%)</td>
                    </tr>
                    <tr style='background:#0f172a;border-bottom:1px solid #334155;'>
                      <td style='padding:10px 14px;font-weight:600;'>Failure Instances</td>
                      <td style='padding:10px 14px;'>339 instances (3.39% severe class imbalance)</td>
                    </tr>
                    <tr style='border-bottom:1px solid #334155;'>
                      <td style='padding:10px 14px;font-weight:600;'>Partitioning Scheme</td>
                      <td style='padding:10px 14px;'>80% Training / 20% Testing (Stratified by failure status)</td>
                    </tr>
                    <tr style='background:#0f172a;'>
                      <td style='padding:10px 14px;font-weight:600;'>Operational Input Features</td>
                      <td style='padding:10px 14px;'>6 (Type, Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            """)

        # ══════════════════════════════════════════════════════════════════
        # TAB 4 – About
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("ℹ️ About"):
            gr.HTML("""
            <div style='max-width:880px;margin:0 auto;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;'>

              <!-- Title banner -->
              <div style='background:linear-gradient(135deg,#0f172a,#1e3a8a);color:white;
                          border-radius:12px;padding:26px;margin-bottom:20px;border:1px solid #2563eb;'>
                <h2 style='margin:0 0 8px 0;font-size:1.5rem;font-weight:800;color:#ffffff;'>
                  Explainable Ensemble Learning for Industrial Predictive Maintenance
                </h2>
                <p style='margin:0;color:#93c5fd;font-size:1rem;font-weight:500;'>
                  Real-time Prognostics and Explainable AI (SHAP) with LightGBM
                </p>
              </div>

              <!-- Abstract -->
              <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
                          padding:20px;margin-bottom:20px;'>
                <h3 style='color:#38bdf8;margin:0 0 10px 0;font-size:1.2rem;font-weight:700;'>📄 Executive Summary</h3>
                <p style='color:#e2e8f0;line-height:1.7;margin:0;font-size:0.95rem;'>
                  In Industry 4.0 manufacturing environments, unexpected equipment failures lead to massive downtime
                  and economic loss. This application implements an end-to-end predictive maintenance system
                  combining a highly tuned <b>LightGBM</b> gradient boosting classifier with <b>SHAP</b>
                  (SHapley Additive exPlanations). Evaluated on the UCI AI4I 2020 dataset (10,000 records, 3.39% failure rate),
                  the model achieves <b>97.85% accuracy</b>, an <b>F1-score of 0.7514</b>, and a <b>recall of 95.59%</b>
                  (capturing 65 of 68 test failure events). TreeExplainer attribution pinpoints Torque, Tool Wear,
                  and Process Heat Dissipation as leading operational drivers of mechanical breakdown.
                </p>
              </div>

              <!-- Research team -->
              <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
                          padding:20px;margin-bottom:20px;'>
                <h3 style='color:#38bdf8;margin:0 0 16px 0;font-size:1.2rem;font-weight:700;'>👥 Engineering &amp; Research Team</h3>
                <div style='display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:14px;'>
                  <div style='background:#0f172a;border-radius:8px;padding:16px;text-align:center;
                              border:1px solid #334155;'>
                    <div style='font-size:2rem;margin-bottom:6px;'>👨‍💻</div>
                    <b style='color:#f8fafc;font-size:1.05rem;'>Vibhor Kumar</b>
                    <p style='color:#94a3b8;font-size:0.85rem;margin:4px 0 0 0;'>Computer Science &amp; Engineering</p>
                  </div>
                  <div style='background:#0f172a;border-radius:8px;padding:16px;text-align:center;
                              border:1px solid #334155;'>
                    <div style='font-size:2rem;margin-bottom:6px;'>👨‍💻</div>
                    <b style='color:#f8fafc;font-size:1.05rem;'>Rahul Shankar Panday</b>
                    <p style='color:#94a3b8;font-size:0.85rem;margin:4px 0 0 0;'>Computer Science &amp; Engineering</p>
                  </div>
                  <div style='background:#0f172a;border-radius:8px;padding:16px;text-align:center;
                              border:1px solid #334155;'>
                    <div style='font-size:2rem;margin-bottom:6px;'>👨‍💻</div>
                    <b style='color:#f8fafc;font-size:1.05rem;'>Ridesh Rajesh</b>
                    <p style='color:#94a3b8;font-size:0.85rem;margin:4px 0 0 0;'>Computer Science &amp; Engineering</p>
                  </div>
                </div>
                <div style='margin-top:16px;text-align:center;'>
                  <span style='background:#0f172a;color:#93c5fd;padding:8px 20px;
                               border-radius:999px;font-weight:600;font-size:0.92rem;border:1px solid #1e3a8a;display:inline-block;'>
                    🏛️ Department of Computer Science &amp; Engineering<br>
                    RV Institute of Technology and Management, Bengaluru, Karnataka, India
                  </span>
                </div>
              </div>

              <!-- External references -->
              <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
                          padding:20px;margin-bottom:20px;'>
                <h3 style='color:#38bdf8;margin:0 0 12px 0;font-size:1.2rem;font-weight:700;'>🔗 Scientific References &amp; Resources</h3>
                <ul style='margin:0;padding-left:20px;color:#cbd5e1;line-height:1.8;font-size:0.92rem;'>
                  <li>
                    <b>UCI Benchmark Dataset:</b>
                    <a href='https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset'
                       style='color:#38bdf8;text-decoration:none;font-weight:600;' target='_blank'>
                      AI4I 2020 Predictive Maintenance Dataset ↗
                    </a>
                  </li>
                  <li>
                    <b>Explainable AI Framework:</b>
                    <a href='https://shap.readthedocs.io' style='color:#38bdf8;text-decoration:none;font-weight:600;' target='_blank'>
                      SHAP (SHapley Additive exPlanations) Documentation ↗
                    </a>
                  </li>
                  <li>
                    <b>Gradient Boosting Framework:</b>
                    <a href='https://lightgbm.readthedocs.io' style='color:#38bdf8;text-decoration:none;font-weight:600;' target='_blank'>
                      LightGBM (Light Gradient Boosting Machine) ↗
                    </a>
                  </li>
                  <li>
                    <b>Live Demonstration Space:</b>
                    <a href='https://huggingface.co/spaces/iamvk09/predictive-maintenance-lgbm'
                       style='color:#38bdf8;text-decoration:none;font-weight:600;' target='_blank'>
                      Hugging Face Spaces Live App ↗
                    </a>
                  </li>
                  <li>
                    <b>GitHub Source Repository:</b>
                    <a href='https://github.com/iamvk09/Industrial-Predictive-Maintenance'
                       style='color:#38bdf8;text-decoration:none;font-weight:600;' target='_blank'>
                      GitHub Repository iamvk09/Industrial-Predictive-Maintenance ↗
                    </a>
                  </li>
                </ul>
              </div>

              <!-- UN SDG 9 Alignment -->
              <div style='background:rgba(16, 185, 129, 0.12);border:1px solid #10b981;border-radius:10px;
                          padding:18px;'>
                <h4 style='color:#34d399;margin:0 0 8px 0;font-size:1.05rem;font-weight:700;'>🌱 UN Sustainable Development Goal 9 Alignment</h4>
                <p style='color:#e2e8f0;font-size:0.92rem;line-height:1.6;margin:0;'>
                  This system advances <b>UN SDG 9 (Industry, Innovation, and Infrastructure)</b> by optimizing
                  industrial asset longevity, decreasing catastrophic mechanical failures, reducing emergency component
                  replacement waste, and supporting smart manufacturing resilience.
                </p>
              </div>

            </div>
            """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
