"""
ui/app.py
Streamlit frontend for the ECG Arrhythmia Detection API.
Run with: streamlit run ui/app.py
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="ECG Arrhythmia Detector",
    page_icon="❤️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .metric-value { font-size: 2.5rem; font-weight: 700; font-family: 'Space Mono', monospace; }
    .metric-label { font-size: 0.85rem; opacity: 0.7; letter-spacing: 0.1em; text-transform: uppercase; }

    .alert-arrhythmia {
        background: linear-gradient(135deg, #3d0000, #7a0000);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 20px;
        color: #ff8888;
        font-family: 'Space Mono', monospace;
        text-align: center;
        font-size: 1.3rem;
    }
    .alert-normal {
        background: linear-gradient(135deg, #003d00, #007a00);
        border: 2px solid #44ff44;
        border-radius: 12px;
        padding: 20px;
        color: #88ff88;
        font-family: 'Space Mono', monospace;
        text-align: center;
        font-size: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("❤️ ECG Arrhythmia Detector")
st.markdown("**1D CNN model** trained on the MIT-BIH Arrhythmia Database — powered by FastAPI")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
api_base = st.sidebar.text_input("API Base URL", value=API_BASE)

# Health check
with st.sidebar:
    st.markdown("---")
    st.subheader("🔌 API Status")
    try:
        r = requests.get(f"{api_base}/health", timeout=3)
        info = r.json()
        if info.get("status") == "ok":
            model_status = "✅ Loaded" if info.get("model_loaded") else "⚠️ Not loaded (run train.py)"
            st.success(f"API Online\nModel: {model_status}")
        else:
            st.error("API returned unexpected status")
    except Exception:
        st.error("❌ Cannot reach API.\nMake sure FastAPI is running.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📡 Analyze MIT-BIH Record", "📤 Upload Custom Segments"])


# ── TAB 1: Record-based analysis ──────────────────────────────────────────────
with tab1:
    st.subheader("Analyze a MIT-BIH Record")
    st.caption("The API will fetch the record from PhysioNet and classify every beat.")

    col1, col2 = st.columns([2, 1])
    with col1:
        record_id = st.text_input("Record ID", value="106",
                                   help="Any MIT-BIH record (100-234). Common ones: 100, 106, 200, 201.")
    with col2:
        window_size = st.slider("Window Size (samples)", 50, 300, 100,
                                 help="Samples before and after each R-peak.")

    if st.button("🔍 Analyze Record", type="primary", use_container_width=True):
        with st.spinner(f"Fetching and analyzing record {record_id} …"):
            try:
                resp = requests.post(
                    f"{api_base}/predict/record",
                    json={"record_id": record_id, "window_size": window_size},
                    timeout=60,
                )
                if resp.status_code != 200:
                    st.error(f"API Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
                else:
                    result = resp.json()
                    summary = result["summary"]
                    beats = result["beat_predictions"]

                    # ── Summary metrics
                    st.markdown("### 📊 Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    metrics = [
                        ("Total Beats", summary["total_beats"]),
                        ("Abnormal Beats", summary["abnormal_beats"]),
                        ("Abnormal %", f"{summary['abnormal_pct']}%"),
                        ("Threshold", "10%"),
                    ]
                    for col, (label, val) in zip([c1, c2, c3, c4], metrics):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{val}</div>
                                <div class="metric-label">{label}</div>
                            </div>""", unsafe_allow_html=True)

                    # ── Diagnosis banner
                    st.markdown("### 🩺 Diagnosis")
                    if summary["arrhythmia_detected"]:
                        st.markdown('<div class="alert-arrhythmia">⚠️ ARRHYTHMIA DETECTED</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-normal">✅ NORMAL ECG</div>',
                                    unsafe_allow_html=True)

                    # ── Beat-level bar chart
                    st.markdown("### 📈 Beat-Level Predictions")
                    df = pd.DataFrame(beats)
                    normal_count = (df["label_code"] == 0).sum()
                    abnormal_count = (df["label_code"] == 1).sum()

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                             facecolor='#0d1117')
                    for ax in axes:
                        ax.set_facecolor('#161b22')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#30363d')

                    # Pie chart
                    axes[0].pie(
                        [normal_count, abnormal_count],
                        labels=["Normal", "Abnormal"],
                        colors=["#22c55e", "#ef4444"],
                        autopct='%1.1f%%',
                        textprops={"color": "white", "fontsize": 12},
                        startangle=90,
                    )
                    axes[0].set_title("Beat Distribution", color="white", fontsize=13)

                    # Beat sequence scatter
                    colors = ["#22c55e" if l == 0 else "#ef4444" for l in df["label_code"]]
                    axes[1].scatter(df["beat_index"], df["label_code"],
                                    c=colors, s=15, alpha=0.7)
                    axes[1].set_yticks([0, 1])
                    axes[1].set_yticklabels(["Normal", "Abnormal"], color="white")
                    axes[1].set_xlabel("Beat Index", color="white")
                    axes[1].set_title("Beat Sequence", color="white", fontsize=13)
                    axes[1].tick_params(colors="white")

                    normal_patch = mpatches.Patch(color="#22c55e", label="Normal")
                    abnormal_patch = mpatches.Patch(color="#ef4444", label="Abnormal")
                    axes[1].legend(handles=[normal_patch, abnormal_patch],
                                   facecolor="#161b22", labelcolor="white")

                    plt.tight_layout()
                    st.pyplot(fig)

                    # ── Data table
                    with st.expander("📋 View all beat predictions"):
                        st.dataframe(df, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to API. Is FastAPI running?")


# ── TAB 2: Custom segment upload ─────────────────────────────────────────────
with tab2:
    st.subheader("Upload Custom Beat Segments")
    st.caption("Upload a CSV where each **row** is a 200-sample beat segment (no header).")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded, header=None).values.astype(float)
            st.info(f"Loaded {data.shape[0]} segments × {data.shape[1]} samples")

            if data.shape[1] != 200:
                st.error(f"Each segment must have exactly 200 samples (window_size=100). Got {data.shape[1]}.")
            else:
                # Preview first beat
                st.markdown("#### Preview — Beat #0")
                fig, ax = plt.subplots(figsize=(10, 2.5), facecolor='#0d1117')
                ax.set_facecolor('#161b22')
                ax.plot(data[0], color="#60a5fa", linewidth=1.2)
                ax.set_title("Beat Segment", color="white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor('#30363d')
                st.pyplot(fig)

                if st.button("🔍 Classify Segments", type="primary", use_container_width=True):
                    with st.spinner("Classifying …"):
                        try:
                            resp = requests.post(
                                f"{api_base}/predict",
                                json={"segments": data.tolist()},
                                timeout=60,
                            )
                            if resp.status_code != 200:
                                st.error(f"API Error: {resp.json().get('detail', resp.text)}")
                            else:
                                result = resp.json()
                                summary = result["summary"]

                                st.markdown("### 📊 Results")
                                c1, c2, c3 = st.columns(3)
                                for col, (label, val) in zip(
                                    [c1, c2, c3],
                                    [("Total Beats", summary["total_beats"]),
                                     ("Abnormal Beats", summary["abnormal_beats"]),
                                     ("Abnormal %", f"{summary['abnormal_pct']}%")]
                                ):
                                    with col:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{val}</div>
                                            <div class="metric-label">{label}</div>
                                        </div>""", unsafe_allow_html=True)

                                if summary["arrhythmia_detected"]:
                                    st.markdown('<div class="alert-arrhythmia">⚠️ ARRHYTHMIA DETECTED</div>',
                                                unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="alert-normal">✅ NORMAL ECG</div>',
                                                unsafe_allow_html=True)

                                df_res = pd.DataFrame(result["beat_predictions"])
                                st.dataframe(df_res, use_container_width=True)

                        except requests.exceptions.ConnectionError:
                            st.error("❌ Could not connect to API.")

        except Exception as e:
            st.error(f"Error reading file: {e}")
