import os
import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer

# ==========================================================
# PAGE CONFIG (must be first Streamlit command)
# ==========================================================

st.set_page_config(
    page_title="Automotive Defect Triage AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# ==========================================================
# GLOBAL STYLE
# ==========================================================

st.markdown("""
<style>

    .main {
        background-color: #f5f7fa;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-top:20px;
        color: #1f2937;
        margin-bottom: 0px;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0px;
        margin-bottom: 1.5rem;
    }

    .section-card {
        background-color: #ffffff;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.75rem;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #6b7280;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
        border: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #f3f4f6 !important;
    }

</style>
""", unsafe_allow_html=True)



def section_start(title):
    st.markdown(f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
    """, unsafe_allow_html=True)


def section_end():
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# LOAD MODEL ARTIFACTS
# ==========================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("catboost_defect_triage_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder


model, encoder = load_artifacts()

REQUIRED_COLUMNS = [
    "Title",
    "Description",
    "Module",
    "Subsystem",
    "Severity",
    "Priority",
    "Vehicle_Model",
    "Variant",
    "Software_Version"
]

HISTORY_FILE = "prediction_history.xlsx"
COMPARISON_FILE = "prediction_comparison.xlsx"
FEATURE_IMPORTANCE_FILE = "feature_importance.xlsx"
LOG_FILE = "vehicle.log"


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def parse_log(log_text):
    stats = {
        "CAN_TIMEOUT": len(re.findall(r"CAN_TIMEOUT", log_text)),
        "UDS_NEG_RESPONSE": len(re.findall(r"UDS_NEG_RESPONSE", log_text)),
        "DISPLAY_FREEZE": len(re.findall(r"DISPLAY_FREEZE", log_text)),
        "FIRMWARE_TASK_FAILURE": len(re.findall(r"FIRMWARE_TASK_FAILURE", log_text)),
        "FLASH_FAILURE": len(re.findall(r"FLASH_FAILURE", log_text)),
        "CRC_ERROR": len(re.findall(r"CRC_ERROR", log_text)),
    }

    dtcs = re.findall(r"[PCBU][0-9]{4}", log_text)
    stats["DTC_Count"] = len(dtcs)

    return stats


def determine_team(log_text):
    if "CAN_TIMEOUT" in log_text:
        return "CAN Team"
    if "UDS_NEG_RESPONSE" in log_text:
        return "Diagnostic Team"
    if "DISPLAY_FREEZE" in log_text:
        return "HMI Team"
    if "FIRMWARE_TASK_FAILURE" in log_text:
        return "Embedded Team"
    return "Embedded Team"


def build_features(df):
    df = df.copy()

    df["Issue_Text"] = (
            df["Title"].astype(str) + " " + df["Description"].astype(str)
    )

    df["Module_Subsystem"] = (
            df["Module"].astype(str) + "_" + df["Subsystem"].astype(str)
    )

    df["Severity_Priority"] = (
            df["Severity"].astype(str) + "_" + df["Priority"].astype(str)
    )

    df["Version_Module"] = (
            df["Software_Version"].astype(str) + "_" + df["Module"].astype(str)
    )

    return df


FEATURES = [
    "Issue_Text",
    "Module",
    "Subsystem",
    "Module_Subsystem",
    "Severity",
    "Priority",
    "Severity_Priority",
    "Vehicle_Model",
    "Variant",
    "Software_Version",
    "Version_Module"
]


def safe_read_excel(path):
    if os.path.exists(path):
        return pd.read_excel(path)
    return None


# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div style="margin-top:20px;"></div>
    """,
    unsafe_allow_html=True
)

st.image(
    "Embitel_logo.png",
    width=180
)
st.markdown("""
<style>

.app-title{
    text-align:center;
}

</style>
""", unsafe_allow_html=True)
st.markdown('<div class="app-title"> Automotive Defect Triage AI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">MiniLM + CatBoost-powered intelligent defect routing and analysis system</div>',
            unsafe_allow_html=True)
st.markdown("""
<style>

.app-sub{
    margin-top:0px;
    color:black;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">An automotive predictive defect triage system that analyze open defects to predict the correct responsible engineer across global teams in the Project.</div>',
            unsafe_allow_html=True)

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:
    st.markdown("## Navigation")
    menu = st.radio(
        "",
        [
            "📊 Summary",
            "📁 Batch Prediction",
            "🕘 Prediction History",
            "🎯 Model Evaluation",
            "📈 Feature Importance",
            "🧾 Log Parser"

        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model:** MiniLM + CatBoost Classifier")

# ==========================================================
# DASHBOARD PAGE
# ==========================================================

if menu == "📊 Summary":

    section_start("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "MiniLM + CatBoost")
    col2.metric("Classes", len(encoder.classes_))
    col3.metric("Attributes", "11")
    col4.metric("Status", "Ready ✅")
    section_end()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        section_start("Model Overview")
        st.markdown("""
        **Objective**
        Predict the responsible engineering team for an automotive defect fixes.

        **Algorithm**
        - MiniLM Embeddings
        - CatBoost Classifier
        - Hybrid AI Architecture
        - Multi-Class Classification
        - Supervised Learning
        """)
        section_end()

        section_start("Training Configuration")
        config_df = pd.DataFrame({
            "Parameter": [
                "Algorithm", "Iterations", "Depth",
                "Learning Rate", "Loss Function",
                "Evaluation Metric", "Train/Test Split"
            ],
            "Value": [
                "MiniLM + CatBoost", "1000", "8",
                "0.05", "MultiClass",
                "Accuracy", "80/20"
            ]
        })
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        section_end()

    with col_right:
        section_start("Model Workflow")
        st.code("""
    Defect Dataset
          ↓
    Feature Engineering
          ↓
    Issue_Text
          ↓
    MiniLM Embedding Model
          ↓
    Semantic Embeddings
          +
    Automotive Metadata
          ↓
    CatBoost Classifier
          ↓
    Team Prediction
          ↓
    Confidence Score
        """)
        section_end()

        section_start("Target Teams")
        team_df = pd.DataFrame({"Team": encoder.classes_})
        st.dataframe(team_df, use_container_width=True, hide_index=True)
        section_end()

    comparison = safe_read_excel(COMPARISON_FILE)

    if comparison is not None:
        section_start("Identified Team Distribution")
        team_count = comparison["Predicted_Team"].value_counts().reset_index()
        team_count.columns = ["Team", "Count"]

        fig = px.pie(
            team_count,
            names="Team",
            values="Count",
            hole=0.45
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        section_end()


# ==========================================================
# BATCH PREDICTION PAGE
# ==========================================================

elif menu == "📁 Batch Prediction":

    section_start("Upload Defect Data")
    uploaded_excel = st.file_uploader("Upload Defect Excel (.xlsx)", type=["xlsx"])
    section_end()

    if uploaded_excel:

        df = pd.read_excel(uploaded_excel)

        section_start("Input Preview")
        st.dataframe(df.head(), use_container_width=True)
        section_end()

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            df = build_features(df)

            run_col, _ = st.columns([1, 4])
            run = run_col.button("🚀 Run Batch Prediction", use_container_width=True)

            if run:
                structured_features = [

                    "Module",
                    "Subsystem",
                    "Module_Subsystem",
                    "Severity",
                    "Priority",
                    "Severity_Priority",
                    "Vehicle_Model",
                    "Variant",
                    "Software_Version",
                    "Version_Module"

                ]

                embeddings = embedding_model.encode(
                    df["Issue_Text"].tolist()
                )

                embedding_df = pd.DataFrame(
                    embeddings,
                    columns=[
                        f"emb_{i}"
                        for i in range(
                            embeddings.shape[1]
                        )
                    ]
                )

                X = pd.concat(
                    [
                        embedding_df,
                        df[
                            structured_features
                        ].reset_index(drop=True)
                    ],
                    axis=1
                )

                predictions = model.predict(X)
                probabilities = model.predict_proba(X)

                predicted_teams = encoder.inverse_transform(
                    predictions.flatten().astype(int)
                )
                confidence = probabilities.max(axis=1)

                result_df = df.copy()
                result_df["Predicted_Team"] = predicted_teams
                result_df["Confidence"] = np.round(confidence, 4)

                old_history = safe_read_excel(HISTORY_FILE)
                history_df = (
                    pd.concat([old_history, result_df], ignore_index=True)
                    if old_history is not None else result_df.copy()
                )
                history_df.to_excel(HISTORY_FILE, index=False)

                for _, row in result_df.iterrows():
                    st.session_state.history.append({
                        "Title": row["Title"],
                        "Predicted_Team": row["Predicted_Team"],
                        "Confidence": row["Confidence"]
                    })

                section_start(f"Results — {len(result_df)} Records Processed")
                st.dataframe(result_df, use_container_width=True)
                section_end()

                col_a, col_b = st.columns([1, 1])

                with col_a:
                    section_start("Predicted Team Distribution")
                    team_count = result_df["Predicted_Team"].value_counts().reset_index()
                    team_count.columns = ["Team", "Count"]
                    fig = px.pie(team_count, names="Team", values="Count", hole=0.45)
                    st.plotly_chart(fig, use_container_width=True)
                    section_end()

                with col_b:
                    section_start("Confidence Distribution")
                    fig2 = px.histogram(result_df, x="Confidence", nbins=20)
                    st.plotly_chart(fig2, use_container_width=True)
                    section_end()

                output_file = "batch_predictions.xlsx"
                result_df.to_excel(output_file, index=False)

                with open(output_file, "rb") as f:
                    st.download_button(
                        "⬇️ Download Results",
                        f,
                        file_name=output_file,
                        use_container_width=True
                    )


# ==========================================================
# MODEL EVALUATION PAGE
# ==========================================================

elif menu == "🎯 Model Evaluation":

    df = safe_read_excel(COMPARISON_FILE)

    if df is None:
        st.warning(f"{COMPARISON_FILE} not found.")
    else:
        total = len(df)
        correct = len(df[df["Correct"] == True])
        accuracy = correct / total if total else 0

        section_start("Performance Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        col2.metric("Total Records", total)
        col3.metric("Correct Predictions", correct)
        section_end()

        tab1, tab2 = st.tabs(["✅ Sample Predictions", "❌ Misclassified Defects"])

        with tab1:
            section_start("Sample Predictions")
            st.dataframe(df.head(50), use_container_width=True)
            section_end()

        with tab2:
            errors = df[df["Correct"] == False]
            section_start(f"Misclassified Defects ({len(errors)})")
            st.dataframe(errors, use_container_width=True)
            section_end()


# ==========================================================
# FEATURE IMPORTANCE PAGE
# ==========================================================

elif menu == "📈 Feature Importance":

    fi = safe_read_excel(FEATURE_IMPORTANCE_FILE)

    if fi is None:
        st.warning(f"{FEATURE_IMPORTANCE_FILE} not found.")
    else:
        section_start("Feature Importance Chart")
        fig = px.bar(
            fi.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        section_end()

        section_start("Feature Importance Table")
        st.dataframe(fi, use_container_width=True, hide_index=True)
        section_end()


# ==========================================================
# LOG PARSER PAGE
# ==========================================================

elif menu == "🧾 Log Parser":

    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        st.error(f"Unable to read {LOG_FILE}: {e}")
        content = ""

    if content:

        section_start("Log Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Log Length", len(content))
        col2.metric("WARNING Logs", len(re.findall(r"WARNING", content)))
        col3.metric("ERROR Logs", len(re.findall(r"ERROR", content)))
        section_end()

        tab1, tab2 = st.tabs(["📄 Raw Log", "🔍 Parsed Analysis"])

        with tab1:
            section_start("Complete Log")
            st.text_area("Vehicle Log", content, height=300, label_visibility="collapsed")
            section_end()

        with tab2:
            accuracy_match = re.search(r"Accuracy\s*[:=]\s*(\d+\.\d+)", content)
            if accuracy_match:
                section_start("Model Accuracy (from log)")
                st.metric("Model Accuracy", f"{float(accuracy_match.group(1)) * 100:.2f}%")
                section_end()

            result = parse_log(content)
            result_df = pd.DataFrame({
                "Error Type": list(result.keys()),
                "Count": list(result.values())
            })

            col_a, col_b = st.columns([1, 1])

            with col_a:
                section_start("Detected Issues")
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                section_end()

            with col_b:
                section_start("Issue Frequency")
                chart = px.bar(result_df, x="Error Type", y="Count", color="Count")
                st.plotly_chart(chart, use_container_width=True)
                section_end()

            predicted_team = determine_team(content)
           # st.success(f"🧠 Likely Responsible Team: **{predicted_team}**")


# ==========================================================
# PREDICTION HISTORY PAGE
# ==========================================================

elif menu == "🕘 Prediction History":

    history_df = safe_read_excel(HISTORY_FILE)

    if history_df is None:
        st.warning(f"{HISTORY_FILE} not found.")
    else:
        section_start("Prediction History Summary")
        col1, col2 = st.columns(2)
        col1.metric("Total Predictions", len(history_df))
        col2.metric("Unique Teams", history_df["Predicted_Team"].nunique())
        section_end()

        section_start("Full History")
        st.dataframe(history_df, use_container_width=True)
        section_end()
