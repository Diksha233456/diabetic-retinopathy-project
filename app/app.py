import streamlit as st
import tempfile
import os

from model.model_loader import load_model
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
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #8b949e;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="title">👁️ Diabetic Retinopathy Detection</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a retinal image and get instant severity prediction</div>',
            unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model("saved_models/model.pth")


model = get_model()


# ── Upload UI ──────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload retinal image",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)


# ── Inference ──────────────────────────────────────────────
if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(uploaded_file.name)[-1],
        delete=False
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("🔍 Analyzing image..."):

            image_array = preprocess(tmp_path)
            predicted_label, probabilities = predict(model, image_array)

        # ── Result Card ─────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_file, use_container_width=True)

        with col2:
            st.subheader("Diagnosis Result")

            if predicted_label == "No Diabetic Retinopathy":
                st.success("✅ No signs of Diabetic Retinopathy detected")
            else:
                st.error(f"⚠️ {predicted_label} detected")
                st.warning(
                    "Possible retinal damage. Consult an ophthalmologist.")

            confidence = max(probabilities.values())
            st.metric("Confidence", f"{confidence*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Probability Card ────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("Detailed Class Probabilities")

        for label in DR_LABELS:
            prob = probabilities[label]
            st.write(label)
            st.progress(prob)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during inference: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.info("📤 Upload a retinal image to begin analysis")
