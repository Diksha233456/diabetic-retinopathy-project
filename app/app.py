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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.card {
    background-color: #161b22;
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0px 4px 24px rgba(0,0,0,0.4);
    margin-bottom: 20px;
    border: 1px solid #21262d;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.subtitle {
    text-align: center;
    color: #8b949e;
    font-size: 16px;
    margin-bottom: 28px;
}
.model-badge {
    text-align: center;
    margin-bottom: 20px;
}
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="title">👁️ Diabetic Retinopathy Detection</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a retinal fundus image to get AI-powered severity classification</div>',
            unsafe_allow_html=True)
st.markdown('<div class="model-badge"><span class="badge">🧠 ResNet-50 Model</span></div>',
            unsafe_allow_html=True)


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
    "No Diabetic Retinopathy":         "🟢",
    "Mild Diabetic Retinopathy":       "🟡",
    "Moderate Diabetic Retinopathy":   "🟠",
    "Severe Diabetic Retinopathy":     "🔴",
    "Proliferative Diabetic Retinopathy": "🔴",
}


# ── Upload UI ──────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
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

            icon = SEVERITY_COLOR.get(predicted_label, "⚪")

            if predicted_label == "No Diabetic Retinopathy":
                st.success(f"{icon} No signs of Diabetic Retinopathy detected")
            elif predicted_label == "Mild Diabetic Retinopathy":
                st.warning(f"{icon} {predicted_label} detected")
                st.info("Early signs present. Schedule a follow-up with an ophthalmologist.")
            else:
                st.error(f"{icon} {predicted_label} detected")
                st.warning("Significant retinal damage detected. Please consult an ophthalmologist immediately.")

            confidence = probabilities[predicted_label]
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Probability Chart ────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Class-wise Probabilities")

        for label in DR_LABELS:
            prob = probabilities.get(label, 0.0)
            icon = SEVERITY_COLOR.get(label, "⚪")
            col_label, col_bar = st.columns([1.5, 3])
            with col_label:
                st.write(f"{icon} **{label}**")
            with col_bar:
                st.progress(float(prob), text=f"{prob * 100:.1f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error during inference: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.info("📤 Upload a retinal fundus image above to begin analysis.")
    st.markdown("""
    **Tips for best results:**
    - Use high-quality fundus/retinal photographs
    - Ensure the optic disc is clearly visible
    - Avoid blurry or over-exposed images
    """)
    st.markdown('</div>', unsafe_allow_html=True)
