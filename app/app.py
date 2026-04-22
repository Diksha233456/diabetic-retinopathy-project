import streamlit as st
import tempfile
import os

from model.model_loader import load_resnet_model
from model.predict import predict, DR_LABELS
from utils.preprocess import preprocess


# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)


# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0B0F1A 0%, #0d1224 50%, #0f1320 100%);
    min-height: 100vh;
}
.main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 860px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}

/* ── Sidebar success message ── */
div[data-testid="stSidebar"] .stAlert {
    background: rgba(34, 197, 94, 0.1) !important;
    border: 1px solid rgba(34, 197, 94, 0.25) !important;
    border-radius: 10px !important;
    color: #86efac !important;
}

/* ── Card base ── */
.card {
    background: linear-gradient(145deg, rgba(22, 27, 46, 0.9), rgba(17, 24, 39, 0.95));
    padding: 28px 32px;
    border-radius: 18px;
    box-shadow: 0 4px 32px rgba(0, 0, 0, 0.45), 0 1px 0 rgba(255,255,255,0.04) inset;
    margin-bottom: 24px;
    border: 1px solid rgba(99, 102, 241, 0.18);
    backdrop-filter: blur(12px);
    transition: box-shadow 0.25s ease;
}
.card:hover {
    box-shadow: 0 8px 48px rgba(99, 102, 241, 0.15), 0 1px 0 rgba(255,255,255,0.04) inset;
}

/* ── Title ── */
.app-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.15;
    margin-bottom: 10px;
}
.app-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 15px;
    font-weight: 400;
    letter-spacing: 0.2px;
    margin-bottom: 20px;
}

/* ── Model badge ── */
.model-badge-wrap {
    text-align: center;
    margin-bottom: 28px;
}
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(139,92,246,0.4);
    color: #c4b5fd;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 100px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

/* ── Divider ── */
.glow-divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(99,102,241,0.4) 50%, transparent 100%);
    margin: 8px 0 24px 0;
    border: none;
}

/* ── Upload zone enhancer ── */
.upload-card {
    background: linear-gradient(145deg, rgba(22, 27, 46, 0.9), rgba(17, 24, 39, 0.95));
    padding: 36px 32px;
    border-radius: 18px;
    border: 2px dashed rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 32px rgba(0,0,0,0.4);
    margin-bottom: 24px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
    text-align: center;
}
.upload-card:hover {
    border-color: rgba(139, 92, 246, 0.55);
    box-shadow: 0 4px 40px rgba(99,102,241,0.12);
}
.upload-icon {
    font-size: 36px;
    margin-bottom: 8px;
}
.upload-label {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 16px;
    font-weight: 500;
}

/* ── Section heading ── */
.section-heading {
    font-size: 18px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Diagnosis subheading ── */
.diagnosis-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.diagnosis-value {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 16px;
}

/* ── Severity pill ── */
.severity-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 14px;
}
.pill-green  { background: rgba(34,197,94,0.15);  border: 1px solid rgba(34,197,94,0.35);  color: #86efac; }
.pill-yellow { background: rgba(234,179,8,0.15);  border: 1px solid rgba(234,179,8,0.35);  color: #fef08a; }
.pill-orange { background: rgba(249,115,22,0.15); border: 1px solid rgba(249,115,22,0.35); color: #fdba74; }
.pill-red    { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.35);  color: #fca5a5; }

/* ── Alert boxes ── */
.alert-info {
    background: rgba(96,165,250,0.08);
    border-left: 3px solid #60a5fa;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 13.5px;
    color: #93c5fd;
    margin-top: 10px;
}
.alert-warn {
    background: rgba(251,191,36,0.08);
    border-left: 3px solid #fbbf24;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 13.5px;
    color: #fcd34d;
    margin-top: 10px;
}
.alert-danger {
    background: rgba(239,68,68,0.08);
    border-left: 3px solid #ef4444;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 13.5px;
    color: #fca5a5;
    margin-top: 10px;
}
.alert-success {
    background: rgba(34,197,94,0.08);
    border-left: 3px solid #22c55e;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 13.5px;
    color: #86efac;
    margin-top: 10px;
}

/* ── Confidence display ── */
.confidence-wrap {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid rgba(99,102,241,0.12);
}
.confidence-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.confidence-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
}
.confidence-pct {
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.confidence-bar-bg {
    width: 100%;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #818cf8, #a78bfa, #60a5fa);
    transition: width 0.5s ease;
    box-shadow: 0 0 10px rgba(139,92,246,0.5);
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.prob-label {
    font-size: 12.5px;
    font-weight: 500;
    color: #94a3b8;
    width: 220px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
}
.prob-bar-green  { background: linear-gradient(90deg, #22c55e, #86efac); box-shadow: 0 0 6px rgba(34,197,94,0.4); }
.prob-bar-yellow { background: linear-gradient(90deg, #eab308, #fef08a); box-shadow: 0 0 6px rgba(234,179,8,0.4); }
.prob-bar-orange { background: linear-gradient(90deg, #f97316, #fdba74); box-shadow: 0 0 6px rgba(249,115,22,0.4); }
.prob-bar-red    { background: linear-gradient(90deg, #ef4444, #fca5a5); box-shadow: 0 0 6px rgba(239,68,68,0.4); }
.prob-bar-deep   { background: linear-gradient(90deg, #dc2626, #f87171); box-shadow: 0 0 6px rgba(220,38,38,0.4); }
.prob-pct {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Tip card ── */
.tip-card {
    background: linear-gradient(145deg, rgba(22,27,46,0.9), rgba(17,24,39,0.95));
    padding: 28px 32px;
    border-radius: 18px;
    border: 1px solid rgba(99,102,241,0.14);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    margin-bottom: 24px;
}
.tip-title {
    font-size: 15px;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.tip-item {
    font-size: 13.5px;
    color: #64748b;
    padding: 5px 0;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}
.tip-dot {
    color: #4f46e5;
    font-size: 16px;
    line-height: 1.3;
}

/* ── Streamlit overrides ── */
div[data-testid="stFileUploader"] {
    border: none !important;
    padding: 0 !important;
}
div[data-testid="stFileUploader"] > div {
    border: 2px dashed rgba(99,102,241,0.3) !important;
    border-radius: 14px !important;
    background: rgba(15,19,36,0.5) !important;
    transition: border-color 0.25s ease !important;
}
div[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(139,92,246,0.55) !important;
}
div[data-testid="stFileUploader"] label {
    color: #94a3b8 !important;
}
.stSpinner > div {
    border-top-color: #818cf8 !important;
}
/* Progress bar overrides */
div[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, #818cf8, #a78bfa) !important;
}
/* Metric */
div[data-testid="stMetric"] {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-size: 11px !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 28px !important;
    font-weight: 800 !important;
}
/* Subheader */
h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}
/* Image caption */
div[data-testid="stImage"] p {
    color: #64748b !important;
    font-size: 12px !important;
    text-align: center !important;
}
/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0B0F1A; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="app-title">👁️ Diabetic Retinopathy Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload a retinal fundus image to get AI-powered severity classification</div>', unsafe_allow_html=True)
st.markdown('<div class="model-badge-wrap"><span class="model-badge">🧠 ResNet-50 Model</span></div>', unsafe_allow_html=True)
st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_resnet_model("saved_models/dr_model_resnet50.pth")


try:
    model = get_model()
    st.sidebar.success("✅ ResNet-50 model loaded")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ── Sidebar info ───────────────────────────────────────────
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**Model:** ResNet-50 (fine-tuned)  
**Classes:** 5 DR severity levels  
**Input:** Retinal fundus images  
**Input size:** 224 × 224 px  

---

**DR Severity Scale:**
- 🟢 No DR
- 🟡 Mild
- 🟠 Moderate
- 🔴 Severe
- 🔴 Proliferative

---
⚠️ *For educational use only. Not for medical diagnosis.*
""")


# ── Severity colour map ─────────────────────────────────────
SEVERITY_COLOR = {
    "No Diabetic Retinopathy":            "🟢",
    "Mild Diabetic Retinopathy":          "🟡",
    "Moderate Diabetic Retinopathy":      "🟠",
    "Severe Diabetic Retinopathy":        "🔴",
    "Proliferative Diabetic Retinopathy": "🔴",
}

# Pill classes per severity
_PILL_CLASS = {
    "No Diabetic Retinopathy":            "pill-green",
    "Mild Diabetic Retinopathy":          "pill-yellow",
    "Moderate Diabetic Retinopathy":      "pill-orange",
    "Severe Diabetic Retinopathy":        "pill-red",
    "Proliferative Diabetic Retinopathy": "pill-red",
}

# Bar gradient classes per label
_BAR_CLASS = {
    "No Diabetic Retinopathy":            "prob-bar-green",
    "Mild Diabetic Retinopathy":          "prob-bar-yellow",
    "Moderate Diabetic Retinopathy":      "prob-bar-orange",
    "Severe Diabetic Retinopathy":        "prob-bar-red",
    "Proliferative Diabetic Retinopathy": "prob-bar-deep",
}


# ── Upload UI ──────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">📤 Upload Retinal Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)


# ── Inference ──────────────────────────────────────────────
if uploaded_file:

    # Save to temp file
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(uploaded_file.name)[-1],
        delete=False
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("🔍 Analyzing retinal image with ResNet-50..."):
            image_array = preprocess(tmp_path)
            predicted_label, probabilities = predict(model, image_array)

        # ── Result Card ─────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_file, caption="Uploaded fundus image", use_container_width=True)

        with col2:
            st.subheader("Diagnosis Result")

            icon       = SEVERITY_COLOR.get(predicted_label, "⚪")
            pill_cls   = _PILL_CLASS.get(predicted_label, "pill-green")

            # Severity pill
            st.markdown(
                f'<div class="severity-pill {pill_cls}">{icon} {predicted_label}</div>',
                unsafe_allow_html=True
            )

            # Alert message (logic unchanged)
            if predicted_label == "No Diabetic Retinopathy":
                st.markdown(
                    '<div class="alert-success">✅ No signs of Diabetic Retinopathy detected. Your retina appears healthy.</div>',
                    unsafe_allow_html=True
                )
            elif predicted_label == "Mild Diabetic Retinopathy":
                st.markdown(
                    f'<div class="alert-warn">⚠️ {predicted_label} detected.</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="alert-info">🗓️ Early signs present. Schedule a follow-up with an ophthalmologist.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="alert-danger">🚨 {predicted_label} detected.</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="alert-warn">⚠️ Significant retinal damage detected. Please consult an ophthalmologist immediately.</div>',
                    unsafe_allow_html=True
                )

            # Confidence display
            confidence = probabilities[predicted_label]
            confidence_pct = confidence * 100
            st.markdown(f"""
            <div class="confidence-wrap">
                <div class="confidence-label-row">
                    <span class="confidence-label">Confidence Score</span>
                    <span class="confidence-pct">{confidence_pct:.1f}%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{confidence_pct:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Probability Chart ────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">📊 Class-wise Probabilities</div>', unsafe_allow_html=True)

        for label in DR_LABELS:
            prob     = probabilities.get(label, 0.0)
            icon     = SEVERITY_COLOR.get(label, "⚪")
            bar_cls  = _BAR_CLASS.get(label, "prob-bar-green")
            bar_pct  = prob * 100

            st.markdown(f"""
            <div class="prob-row">
                <span class="prob-label">{icon} {label}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill {bar_cls}" style="width:{bar_pct:.1f}%"></div>
                </div>
                <span class="prob-pct">{bar_pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error during inference: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.markdown('<div class="tip-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="tip-title">💡 Tips for best results</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="tip-item"><span class="tip-dot">◆</span> Use high-quality fundus / retinal photographs</div>
    <div class="tip-item"><span class="tip-dot">◆</span> Ensure the optic disc is clearly visible</div>
    <div class="tip-item"><span class="tip-dot">◆</span> Avoid blurry or over-exposed images</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
