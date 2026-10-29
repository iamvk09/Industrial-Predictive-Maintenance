"""
=============================================================================
Explainable Ensemble Learning for Industrial Predictive Maintenance
Using SHAP and LightGBM

Authors: Vibhor Kumar, Rahul Shankar Panday, Ridesh Rajesh
University: RV Institute of Technology and Management, Bengaluru
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
import io
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

# SHAP importance values from paper (used for visualisation when model is absent)
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
        print("✅ Model loaded successfully.")
        return True
    except FileNotFoundError:
        print("⚠️  lgbm_model.pkl not found – running in demo mode.")
        return False
    except Exception as e:
        print(f"⚠️  Model loading error: {e} – running in demo mode.")
        return False

MODEL_LOADED = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_level(prob: float) -> tuple[str, str]:
    """Return (label, hex_colour) for a given failure probability."""
    if prob >= 0.7:
        return "🔴 HIGH RISK",   "#dc2626"
    elif prob >= 0.4:
        return "🟡 MEDIUM RISK", "#d97706"
    else:
        return "🟢 LOW RISK",    "#16a34a"


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
            errors.append(f"<li><b>{feat}</b>: {val} is outside valid range [{lo}, {hi}]</li>")

    if errors:
        html = f"""
        <div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:16px;'>
          <h3 style='color:#dc2626;margin:0 0 8px 0;'>⚠️ Input Validation Error</h3>
          <ul style='margin:0;padding-left:20px;color:#7f1d1d;'>{''.join(errors)}</ul>
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
        # Demo mode: simple heuristic so the UI remains functional
        score = 0.0
        score += max(0, (torque   - 50) / 30) * 0.5
        score += max(0, (tool_wear - 150) / 100) * 0.3
        score += max(0, (2900 - rot_speed) / 1700) * 0.1
        score += (1 - type_val / 2) * 0.1
        prob = min(float(score), 0.99)
        prediction = 1 if prob >= 0.5 else 0

    risk_label, risk_colour = get_risk_level(prob)
    pred_text  = "⚠️ FAILURE PREDICTED" if prediction == 1 else "✅ NORMAL OPERATION"
    pred_colour = "#dc2626" if prediction == 1 else "#16a34a"
    pct = prob * 100

    # ── Result HTML ───────────────────────────────────────────────────────
    html = f"""
    <div style='font-family:Arial,sans-serif;'>
      <!-- Main prediction box -->
      <div style='background:{pred_colour}15;border:2px solid {pred_colour};
                  border-radius:12px;padding:20px;margin-bottom:16px;text-align:center;'>
        <h2 style='color:{pred_colour};margin:0 0 8px 0;font-size:1.6em;'>{pred_text}</h2>
        <p style='margin:0;font-size:1.1em;color:#374151;'>
          Failure Probability: <b style='color:{pred_colour};font-size:1.3em;'>{pct:.1f}%</b>
        </p>
      </div>

      <!-- Risk level -->
      <div style='background:{risk_colour}15;border:1px solid {risk_colour};
                  border-radius:8px;padding:12px;margin-bottom:16px;text-align:center;'>
        <span style='color:{risk_colour};font-size:1.2em;font-weight:bold;'>{risk_label}</span>
      </div>

      <!-- Probability bar -->
      <div style='margin-bottom:16px;'>
        <p style='margin:0 0 4px 0;color:#374151;font-weight:600;'>Failure Probability</p>
        <div style='background:#e5e7eb;border-radius:999px;height:20px;overflow:hidden;'>
          <div style='width:{pct:.1f}%;background:{pred_colour};height:100%;
                      border-radius:999px;transition:width 0.3s ease;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;color:#6b7280;font-size:0.8em;margin-top:2px;'>
          <span>0% (Normal)</span><span>100% (Failure)</span>
        </div>
      </div>

      <!-- Risk thresholds legend -->
      <div style='background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px;'>
        <p style='margin:0 0 8px 0;font-weight:600;color:#374151;'>Risk Thresholds</p>
        <div style='display:flex;gap:16px;flex-wrap:wrap;font-size:0.9em;'>
          <span>🟢 Low Risk: &lt; 40%</span>
          <span>🟡 Medium Risk: 40–70%</span>
          <span>🔴 High Risk: &gt; 70%</span>
        </div>
      </div>

      {"<p style='margin-top:12px;color:#6b7280;font-size:0.85em;font-style:italic;'>⚠️ Running in demo mode (lgbm_model.pkl not found). Upload the model file for real predictions.</p>" if not MODEL_LOADED else ""}
    </div>
    """

    # ── SHAP explanation plot ─────────────────────────────────────────────
    fig = _shap_waterfall_plot(df, prob)
    return html, fig


def _shap_waterfall_plot(df: pd.DataFrame, prob: float) -> plt.Figure:
    """Generate a horizontal SHAP waterfall / bar chart."""
    if MODEL_LOADED and EXPLAINER is not None:
        shap_vals = EXPLAINER.shap_values(df)
        # For binary classifiers shap_values may return list[2]; take class-1
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        else:
            vals = shap_vals[0]
    else:
        # Use paper-reported SHAP values scaled by feature deviation from midpoint
        row = df.iloc[0]
        vals = []
        for feat in FEATURE_NAMES:
            lo, hi = FEATURE_RANGES[feat]
            mid   = (lo + hi) / 2
            scale = (row[feat] - mid) / max(hi - mid, 1)
            vals.append(SHAP_IMPORTANCE[feat] * scale)
        vals = np.array(vals)

    # Sort by absolute value descending
    order   = np.argsort(np.abs(vals))[::-1]
    sorted_feats = [FEATURE_NAMES[i] for i in order]
    sorted_vals  = vals[order]
    colours = ["#dc2626" if v > 0 else "#2563eb" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(sorted_feats[::-1], sorted_vals[::-1], color=colours[::-1],
                   edgecolor="white", height=0.6)

    # Value labels
    for bar, val in zip(bars, sorted_vals[::-1]):
        xpos = bar.get_width() + 0.02 * (1 if val >= 0 else -1)
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=9, color="#374151")

    ax.axvline(0, color="#9ca3af", linewidth=1, linestyle="--")
    ax.set_xlabel("SHAP Value  (impact on failure prediction)", fontsize=9, color="#374151")
    ax.set_title(f"Feature Contributions  |  Failure Prob: {prob*100:.1f}%",
                 fontsize=11, fontweight="bold", color="#1e3a5f", pad=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)

    red_patch  = mpatches.Patch(color="#dc2626", label="↑ Increases failure risk")
    blue_patch = mpatches.Patch(color="#2563eb", label="↓ Decreases failure risk")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")

    fig.patch.set_facecolor("#f9fafb")
    ax.set_facecolor("#f9fafb")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(file_obj):
    """Accept a CSV upload, return (summary_html, result_dataframe, csv_bytes)."""
    if file_obj is None:
        return "<p style='color:#dc2626;'>Please upload a CSV file.</p>", None, None

    try:
        df_raw = pd.read_csv(file_obj.name)
    except Exception as e:
        return f"<p style='color:#dc2626;'>Error reading CSV: {e}</p>", None, None

    # Direct column matching against model's expected names
    col_map = {}
    for need in FEATURE_NAMES:
        # Try exact match first, then case-insensitive
        for col in df_raw.columns:
            if col == need or col.lower() == need.lower():
                col_map[need] = col
                break

    missing = [f for f in FEATURE_NAMES if f not in col_map]
    if missing:
        hint = ", ".join(FEATURE_NAMES)
        return (
            f"<p style='color:#dc2626;'>Missing columns: {', '.join(missing)}.<br>"
            f"Expected columns: <code>{hint}</code></p>",
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
            score += max(0, (row["Torque [Nm]"]    - 50) / 30) * 0.5
            score += max(0, (row["Tool Wear [min]"] - 150) / 100) * 0.3
            score += max(0, (2900 - row["Rotational Speed [rpm]"]) / 1700) * 0.1
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
    fail_rate  = n_fail / total * 100
    avg_prob   = probs.mean() * 100
    high_risk  = int((probs >= 0.7).sum())
    med_risk   = int(((probs >= 0.4) & (probs < 0.7)).sum())
    low_risk   = int((probs < 0.4).sum())

    colour = "#dc2626" if fail_rate > 10 else "#d97706" if fail_rate > 5 else "#16a34a"

    summary_html = f"""
    <div style='font-family:Arial,sans-serif;'>
      <div style='background:#eff6ff;border:2px solid #3b82f6;border-radius:12px;
                  padding:20px;margin-bottom:16px;'>
        <h3 style='color:#1e3a5f;margin:0 0 16px 0;'>📊 Batch Prediction Summary</h3>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;'>
          <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
            <div style='font-size:2em;font-weight:bold;color:#1e3a5f;'>{total}</div>
            <div style='color:#6b7280;font-size:0.9em;'>Total Samples</div>
          </div>
          <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
            <div style='font-size:2em;font-weight:bold;color:#dc2626;'>{n_fail}</div>
            <div style='color:#6b7280;font-size:0.9em;'>Failures Predicted</div>
          </div>
          <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
            <div style='font-size:2em;font-weight:bold;color:#16a34a;'>{n_normal}</div>
            <div style='color:#6b7280;font-size:0.9em;'>Normal Operations</div>
          </div>
          <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
            <div style='font-size:2em;font-weight:bold;color:{colour};'>{fail_rate:.1f}%</div>
            <div style='color:#6b7280;font-size:0.9em;'>Failure Rate</div>
          </div>
          <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
            <div style='font-size:2em;font-weight:bold;color:#7c3aed;'>{avg_prob:.1f}%</div>
            <div style='color:#6b7280;font-size:0.9em;'>Avg Failure Prob</div>
          </div>
          <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
            <div style='font-size:0.95em;font-weight:bold;color:#374151;'>
              🔴 {high_risk} &nbsp; 🟡 {med_risk} &nbsp; 🟢 {low_risk}
            </div>
            <div style='color:#6b7280;font-size:0.9em;'>Risk Distribution</div>
          </div>
        </div>
      </div>
      {"<p style='color:#6b7280;font-style:italic;font-size:0.85em;'>⚠️ Demo mode – upload lgbm_model.pkl for real predictions.</p>" if not MODEL_LOADED else ""}
    </div>
    """

    return summary_html, result_df, result_df.to_csv(index=False)


def download_results(csv_bytes):
    """Return path to a temp CSV file for download."""
    if csv_bytes is None:
        return None
    path = "/tmp/predictions.csv"
    with open(path, "w") as f:
        f.write(csv_bytes)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL INFO FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _model_performance_fig() -> plt.Figure:
    """Bar chart of model comparison metrics from the paper."""
    models = ["RF\n(Baseline)", "SVM\n(Baseline)", "XGBoost", "LightGBM\n★", "Stacking"]
    accuracy = [97.25, 79.10, 97.80, 97.85, 97.75]
    f1_score  = [0.703,  0.226,  0.747,  0.751,  0.743]

    x = np.arange(len(models))
    w = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    bars1 = ax1.bar(x - w/2, accuracy, w, label="Accuracy (%)",
                    color=["#93c5fd"]*2 + ["#3b82f6"]*2 + ["#3b82f6"],
                    edgecolor="white")
    ax1.set_ylabel("Accuracy (%)", color="#1e3a5f", fontsize=10)
    ax1.set_ylim(70, 100)
    ax1.tick_params(axis="y", labelcolor="#1e3a5f")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, f1_score, w, label="F1-Score",
                    color=["#fca5a5"]*2 + ["#ef4444"]*2 + ["#ef4444"],
                    edgecolor="white")
    ax2.set_ylabel("F1-Score", color="#dc2626", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    # Value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color="#1e3a5f")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color="#dc2626")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_title("Model Performance Comparison", fontsize=12, fontweight="bold",
                  color="#1e3a5f", pad=12)

    lines = [
        mpatches.Patch(color="#3b82f6", label="Accuracy (%)"),
        mpatches.Patch(color="#ef4444", label="F1-Score"),
    ]
    ax1.legend(handles=lines, loc="lower right", fontsize=9)

    fig.patch.set_facecolor("#f9fafb")
    ax1.set_facecolor("#f9fafb")
    plt.tight_layout()
    return fig


def _shap_importance_fig() -> plt.Figure:
    """Horizontal bar chart of SHAP feature importances from the paper."""
    feats = [
        "Type", "Air Temperature [K]", "TWF Flag",
        "Process Temp [K]", "OSF Flag", "Rotational Speed",
        "PWF Flag", "HDF Flag", "Tool Wear [min]", "Torque [Nm]",
    ]
    vals = [0.0892, 0.1523, 0.2845, 0.3128, 0.4521, 0.6745, 0.6892, 0.7035, 0.8527, 1.5035]

    cmap = plt.cm.Blues
    colours = [cmap(0.4 + 0.6 * v / max(vals)) for v in vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(feats, vals, color=colours, edgecolor="white", height=0.65)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8.5, color="#374151")

    ax.set_xlabel("Mean |SHAP Value|", fontsize=10, color="#374151")
    ax.set_title("Feature Importance (SHAP Values)", fontsize=12,
                 fontweight="bold", color="#1e3a5f", pad=10)
    ax.set_xlim(0, 1.75)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)

    # Highlight top-3
    for i, (bar, feat) in enumerate(zip(bars, feats)):
        if feat in ("Torque [Nm]", "Tool Wear [min]", "HDF Flag"):
            ax.text(0.02, bar.get_y() + bar.get_height()/2,
                    "★", color="#f59e0b", fontsize=11, va="center")

    fig.patch.set_facecolor("#f9fafb")
    ax.set_facecolor("#f9fafb")
    plt.tight_layout()
    return fig


def _confusion_matrix_fig() -> plt.Figure:
    """Side-by-side confusion matrices for RF vs LightGBM."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    data = [
        ("Random Forest (Baseline)", np.array([[1891, 41], [3, 65]]),  "#93c5fd"),
        ("LightGBM ★ (Proposed)",   np.array([[1905, 27], [3, 65]]),  "#3b82f6"),
    ]

    for ax, (title, cm, colour) in zip(axes, data):
        im = ax.imshow(cm, interpolation="nearest",
                       cmap=plt.cm.Blues if colour == "#3b82f6" else plt.cm.Greys)
        ax.set_title(title, fontsize=9, fontweight="bold", color="#1e3a5f", pad=6)
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Normal", "Failure"], fontsize=8)
        ax.set_yticklabels(["Normal", "Failure"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual",    fontsize=8)

        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "#1e3a5f")

    fig.suptitle("Confusion Matrix Comparison", fontsize=11,
                 fontweight="bold", color="#1e3a5f", y=1.02)
    fig.patch.set_facecolor("#f9fafb")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

# Pre-render static figures
perf_fig = _model_performance_fig()
shap_fig = _shap_importance_fig()
cm_fig   = _confusion_matrix_fig()

# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
/* Global font */
body, .gradio-container { font-family: Arial, sans-serif !important; }

/* Tab header strip */
.tab-nav { background: linear-gradient(135deg, #1e3a5f, #2563eb) !important; }
.tab-nav button { color: white !important; font-weight: 600 !important; }
.tab-nav button.selected { background: rgba(255,255,255,0.2) !important;
                            border-bottom: 3px solid #f59e0b !important; }

/* Section cards */
.gr-box { border-radius: 10px !important; }

/* Inputs */
.gr-number input, .gr-slider input { border-color: #3b82f6 !important; }
"""

# ── Header HTML ──────────────────────────────────────────────────────────────
HEADER_HTML = """
<div style='background:linear-gradient(135deg,#1e3a5f,#2563eb);
            color:white;padding:28px 32px;border-radius:12px;margin-bottom:4px;'>
  <div style='display:flex;align-items:center;gap:16px;'>
    <div style='font-size:3em;'>🏭</div>
    <div>
      <h1 style='margin:0;font-size:1.7em;font-weight:bold;'>
        Industrial Predictive Maintenance
      </h1>
      <p style='margin:4px 0 0 0;font-size:1em;opacity:0.9;'>
        Explainable Ensemble Learning with LightGBM &amp; SHAP
      </p>
    </div>
  </div>
  <div style='margin-top:16px;display:flex;gap:12px;flex-wrap:wrap;'>
    <span style='background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:999px;font-size:0.85em;'>
      📊 Accuracy: 97.85%
    </span>
    <span style='background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:999px;font-size:0.85em;'>
      🎯 F1-Score: 0.7514
    </span>
    <span style='background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:999px;font-size:0.85em;'>
      🔍 Recall: 95.59%
    </span>
    <span style='background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:999px;font-size:0.85em;'>
      🤖 LightGBM + SHAP
    </span>
  </div>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="Predictive Maintenance - RVITM") as demo:

        gr.HTML(HEADER_HTML)

        # ── Status banner ─────────────────────────────────────────────────
        status_colour = "#d1fae5" if MODEL_LOADED else "#fef3c7"
        status_border = "#10b981" if MODEL_LOADED else "#f59e0b"
        status_icon   = "✅" if MODEL_LOADED else "⚠️"
        status_msg    = ("Model loaded successfully – real predictions active."
                         if MODEL_LOADED else
                         "Demo mode – lgbm_model.pkl not found. "
                         "Upload it to the Space files for live predictions.")
        gr.HTML(f"""
        <div style='background:{status_colour};border:1px solid {status_border};
                    border-radius:8px;padding:10px 16px;margin-bottom:8px;font-size:0.9em;'>
          {status_icon} <b>Model Status:</b> {status_msg}
        </div>
        """)

        # ══════════════════════════════════════════════════════════════════
        # TAB 1 – Single Prediction
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🔍 Single Prediction"):
            gr.HTML("<p style='color:#6b7280;margin:0 0 16px 0;'>"
                    "Enter machine parameters below to get a real-time failure prediction "
                    "with SHAP explainability.</p>")

            with gr.Row():
                # ── Left: inputs ─────────────────────────────────────────
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color:#1e3a5f;margin:0 0 8px 0;'>⚙️ Machine Parameters</h3>")

                    type_input = gr.Slider(
                        minimum=0, maximum=2, step=1, value=1,
                        label="Product Type  [0=Low | 1=Medium | 2=High]",
                        info="Quality grade of the product being manufactured",
                    )
                    air_temp_input = gr.Slider(
                        minimum=295, maximum=305, step=0.1, value=300,
                        label="Air Temperature [K]",
                        info="Ambient air temperature (295–305 K)",
                    )
                    proc_temp_input = gr.Slider(
                        minimum=305, maximum=315, step=0.1, value=310,
                        label="Process Temperature [K]",
                        info="Operating process temperature (305–315 K)",
                    )
                    rot_speed_input = gr.Slider(
                        minimum=1200, maximum=2900, step=10, value=1500,
                        label="Rotational Speed [rpm]",
                        info="Machine rotation speed (1200–2900 rpm)",
                    )
                    torque_input = gr.Slider(
                        minimum=3, maximum=80, step=0.5, value=40,
                        label="Torque [Nm]  ⭐ Most Important",
                        info="Torque force – primary failure predictor (3–80 Nm)",
                    )
                    tool_wear_input = gr.Slider(
                        minimum=0, maximum=250, step=1, value=100,
                        label="Tool Wear [min]",
                        info="Cumulative tool usage time (0–250 min)",
                    )

                    with gr.Row():
                        predict_btn = gr.Button("🔮 Predict", variant="primary", scale=2)
                        clear_btn   = gr.Button("🔄 Reset",  variant="secondary", scale=1)

                    # Quick-load examples
                    gr.HTML("<h4 style='color:#1e3a5f;margin:16px 0 6px 0;'>💡 Example Scenarios</h4>")
                    with gr.Row():
                        ex_normal_btn  = gr.Button("✅ Normal Operation",  size="sm")
                        ex_failure_btn = gr.Button("⚠️ High-Risk Scenario", size="sm")

                # ── Right: outputs ────────────────────────────────────────
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color:#1e3a5f;margin:0 0 8px 0;'>📋 Prediction Result</h3>")
                    result_html = gr.HTML()
                    gr.HTML("<h3 style='color:#1e3a5f;margin:16px 0 8px 0;'>🔬 SHAP Explanation</h3>")
                    shap_plot   = gr.Plot(label="Feature Contributions")

            # ── Wiring ───────────────────────────────────────────────────
            inputs  = [type_input, air_temp_input, proc_temp_input,
                       rot_speed_input, torque_input, tool_wear_input]
            outputs = [result_html, shap_plot]

            predict_btn.click(fn=predict_single, inputs=inputs, outputs=outputs)

            clear_btn.click(
                fn=lambda: (1, 300, 310, 1500, 40, 100),
                outputs=[type_input, air_temp_input, proc_temp_input,
                         rot_speed_input, torque_input, tool_wear_input],
            )

            # Normal example: Type=1, Air=300, Proc=310, Speed=1500, Torque=40, Wear=100
            ex_normal_btn.click(
                fn=lambda: (1, 300.0, 310.0, 1500, 40.0, 100),
                outputs=[type_input, air_temp_input, proc_temp_input,
                         rot_speed_input, torque_input, tool_wear_input],
            )
            # High-risk example: Type=0, Air=302, Proc=312, Speed=1200, Torque=75, Wear=220
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
            <p style='color:#6b7280;margin:0 0 8px 0;'>
              Upload a CSV file with multiple machine readings for bulk prediction.
            </p>
            <div style='background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;
                        padding:12px;margin-bottom:16px;font-size:0.9em;'>
              <b>📋 Required CSV columns (exact names):</b><br>
              <code>Type, Air_temp, Process_temp, Rotation_speed, Torque, Tool_wear</code>
            </div>
            """)

            with gr.Row():
                file_input    = gr.File(label="📁 Upload CSV", file_types=[".csv"])
                batch_run_btn = gr.Button("⚡ Run Batch Prediction", variant="primary")

            batch_summary = gr.HTML()

            with gr.Row():
                result_table  = gr.Dataframe(label="Prediction Results", interactive=False)

            with gr.Row():
                csv_state     = gr.State()
                download_btn  = gr.Button("⬇️ Download Results CSV", variant="secondary")
                download_file = gr.File(label="Download", visible=False)

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
            <div style='background:#eff6ff;border:1px solid #93c5fd;border-radius:10px;
                        padding:20px;margin-bottom:16px;'>
              <h3 style='color:#1e3a5f;margin:0 0 12px 0;'>🤖 LightGBM Model Overview</h3>
              <div style='display:grid;grid-template-columns:repeat(2,1fr);gap:12px;'>
                <div>
                  <b style='color:#374151;'>Algorithm:</b>
                  <span style='color:#6b7280;'> LightGBM (Gradient Boosting)</span><br>
                  <b style='color:#374151;'>Trees:</b>
                  <span style='color:#6b7280;'> 100 estimators, depth 6</span><br>
                  <b style='color:#374151;'>Learning Rate:</b>
                  <span style='color:#6b7280;'> 0.1</span><br>
                  <b style='color:#374151;'>Imbalance Fix:</b>
                  <span style='color:#6b7280;'> SMOTE (1:1 ratio)</span>
                </div>
                <div>
                  <b style='color:#374151;'>Accuracy:</b>
                  <span style='color:#16a34a;font-weight:bold;'> 97.85%</span><br>
                  <b style='color:#374151;'>F1-Score:</b>
                  <span style='color:#16a34a;font-weight:bold;'> 0.7514</span><br>
                  <b style='color:#374151;'>Recall:</b>
                  <span style='color:#16a34a;font-weight:bold;'> 95.59%</span><br>
                  <b style='color:#374151;'>Precision:</b>
                  <span style='color:#16a34a;font-weight:bold;'> 0.619</span>
                </div>
              </div>
            </div>
            """)

            with gr.Row():
                gr.Plot(value=perf_fig, label="Model Performance Comparison")
                gr.Plot(value=shap_fig, label="SHAP Feature Importance")

            gr.Plot(value=cm_fig, label="Confusion Matrices: RF vs LightGBM")

            gr.HTML("""
            <div style='background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;
                        padding:20px;margin-top:16px;'>
              <h4 style='color:#1e3a5f;margin:0 0 10px 0;'>📁 Dataset Information</h4>
              <table style='width:100%;border-collapse:collapse;font-size:0.9em;'>
                <tr style='background:#eff6ff;'>
                  <th style='padding:8px;text-align:left;border:1px solid #dbeafe;'>Property</th>
                  <th style='padding:8px;text-align:left;border:1px solid #dbeafe;'>Value</th>
                </tr>
                <tr><td style='padding:8px;border:1px solid #e5e7eb;'>Dataset</td>
                    <td style='padding:8px;border:1px solid #e5e7eb;'>AI4I 2020 Predictive Maintenance</td></tr>
                <tr style='background:#f9fafb;'>
                  <td style='padding:8px;border:1px solid #e5e7eb;'>Total Samples</td>
                  <td style='padding:8px;border:1px solid #e5e7eb;'>10,000</td></tr>
                <tr><td style='padding:8px;border:1px solid #e5e7eb;'>Normal Operations</td>
                    <td style='padding:8px;border:1px solid #e5e7eb;'>9,661 (96.61%)</td></tr>
                <tr style='background:#f9fafb;'>
                  <td style='padding:8px;border:1px solid #e5e7eb;'>Failure Cases</td>
                  <td style='padding:8px;border:1px solid #e5e7eb;'>339 (3.39%)</td></tr>
                <tr><td style='padding:8px;border:1px solid #e5e7eb;'>Train / Test Split</td>
                    <td style='padding:8px;border:1px solid #e5e7eb;'>80% / 20% (stratified)</td></tr>
                <tr style='background:#f9fafb;'>
                  <td style='padding:8px;border:1px solid #e5e7eb;'>Input Features</td>
                  <td style='padding:8px;border:1px solid #e5e7eb;'>6</td></tr>
              </table>
            </div>
            """)

        # ══════════════════════════════════════════════════════════════════
        # TAB 4 – About
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("ℹ️ About"):
            gr.HTML("""
            <div style='max-width:800px;margin:0 auto;font-family:Arial,sans-serif;'>

              <!-- Title card -->
              <div style='background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;
                          border-radius:12px;padding:28px;margin-bottom:24px;'>
                <h2 style='margin:0 0 8px 0;font-size:1.5em;'>
                  Explainable Ensemble Learning for Industrial Predictive Maintenance
                </h2>
                <p style='margin:0;opacity:0.9;font-size:0.95em;'>
                  Using SHAP and LightGBM
                </p>
              </div>

              <!-- Abstract -->
              <div style='background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;
                          padding:20px;margin-bottom:20px;'>
                <h3 style='color:#1e3a5f;margin:0 0 10px 0;'>📄 Abstract</h3>
                <p style='color:#374151;line-height:1.6;margin:0;'>
                  In Industry 4.0 manufacturing environments, unexpected machine failures lead to
                  substantial economic losses. This work introduces a framework combining ensemble
                  machine learning (LightGBM) with SHAP explainability to predict equipment failures
                  from sensor data. Using the AI4I 2020 dataset (10,000 samples, severe 96.6:3.4
                  class imbalance), SMOTE oversampling and LightGBM achieve 97.85% accuracy and
                  0.7514 F1-score – a 5.71% improvement over the Random Forest baseline.
                  SHAP analysis reveals Torque, Tool Wear, and Heat Dissipation as critical
                  failure predictors.
                </p>
              </div>

              <!-- Team -->
              <div style='background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;
                          padding:20px;margin-bottom:20px;'>
                <h3 style='color:#1e3a5f;margin:0 0 16px 0;'>👥 Research Team</h3>
                <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;'>
                  <div style='background:white;border-radius:8px;padding:16px;text-align:center;
                              border:1px solid #dbeafe;'>
                    <div style='font-size:2em;'>👨‍💻</div>
                    <b style='color:#1e3a5f;'>Vibhor Kumar</b>
                    <p style='color:#6b7280;font-size:0.85em;margin:4px 0 0 0;'>6th Semester CSE</p>
                  </div>
                  <div style='background:white;border-radius:8px;padding:16px;text-align:center;
                              border:1px solid #dbeafe;'>
                    <div style='font-size:2em;'>👨‍💻</div>
                    <b style='color:#1e3a5f;'>Rahul Shankar Panday</b>
                    <p style='color:#6b7280;font-size:0.85em;margin:4px 0 0 0;'>6th Semester CSE</p>
                  </div>
                  <div style='background:white;border-radius:8px;padding:16px;text-align:center;
                              border:1px solid #dbeafe;'>
                    <div style='font-size:2em;'>👨‍💻</div>
                    <b style='color:#1e3a5f;'>Ridesh Rajesh</b>
                    <p style='color:#6b7280;font-size:0.85em;margin:4px 0 0 0;'>6th Semester CSE</p>
                  </div>
                </div>
                <div style='margin-top:12px;text-align:center;'>
                  <span style='background:#eff6ff;color:#1e3a5f;padding:6px 16px;
                               border-radius:999px;font-weight:600;font-size:0.9em;'>
                    🏛️ Department of Computer Science &amp; Engineering<br>
                    RV Institute of Technology and Management, Bengaluru, Karnataka, India
                  </span>
                </div>
              </div>

              <!-- Links -->
              <div style='background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;
                          padding:20px;margin-bottom:20px;'>
                <h3 style='color:#1e3a5f;margin:0 0 12px 0;'>🔗 Resources</h3>
                <p style='color:#374151;margin:4px 0;'>
                  📊 <b>Dataset:</b>
                  <a href='https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset'
                     style='color:#2563eb;' target='_blank'>
                    AI4I 2020 – UCI ML Repository
                  </a>
                </p>
                <p style='color:#374151;margin:4px 0;'>
                  📚 <b>SHAP Library:</b>
                  <a href='https://shap.readthedocs.io' style='color:#2563eb;' target='_blank'>
                    shap.readthedocs.io
                  </a>
                </p>
                <p style='color:#374151;margin:4px 0;'>
                  🤖 <b>LightGBM:</b>
                  <a href='https://lightgbm.readthedocs.io' style='color:#2563eb;' target='_blank'>
                    lightgbm.readthedocs.io
                  </a>
                </p>
              </div>

              <!-- SDG -->
              <div style='background:#ecfdf5;border:1px solid #6ee7b7;border-radius:10px;
                          padding:16px;'>
                <h4 style='color:#065f46;margin:0 0 6px 0;'>🌱 SDG 9 Alignment</h4>
                <p style='color:#374151;font-size:0.9em;margin:0;'>
                  This research advances <b>UN Sustainable Development Goal 9</b>
                  (Industry, Innovation and Infrastructure) by enabling data-driven
                  maintenance that reduces industrial waste, prevents costly equipment
                  failures, and promotes sustainable manufacturing practices.
                </p>
              </div>

            </div>
            """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
