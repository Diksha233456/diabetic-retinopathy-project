import streamlit as st
import tempfile
import os
import plotly.graph_objects as go

from model.model_loader import load_resnet_model
from model.predict import predict, DR_LABELS
from utils.preprocess import preprocess


# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Session state ────────────────────────────────────────────────────────────────
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ── CSS ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ══ HIDE SIDEBAR ENTIRELY ══ */
section[data-testid="stSidebar"]  { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }

/* ══ BASE ══ */
*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* ══ BACKGROUND ══ */
.stApp {
    background: radial-gradient(ellipse 130% 60% at 50% -10%, #141b3e 0%, #0B0F1A 55%, #07090f 100%);
    min-height: 100vh;
}
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 5rem !important;
    max-width: 880px !important;
}

/* ══ HERO ══ */
.hero {
    text-align: center;
    padding: 0 0 44px;
}
.hero-ring {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 76px;
    height: 76px;
    border-radius: 50%;
    background: radial-gradient(circle at 40% 40%, rgba(99,102,241,0.35), rgba(0,0,0,0) 70%),
                linear-gradient(135deg, rgba(99,102,241,0.28), rgba(139,92,246,0.28));
    border: 1.5px solid rgba(139,92,246,0.45);
    box-shadow: 0 0 28px rgba(99,102,241,0.35), 0 0 70px rgba(99,102,241,0.12);
    font-size: 36px;
    margin-bottom: 24px;
    animation: pulseRing 3s ease-in-out infinite;
}
@keyframes pulseRing {
    0%, 100% { box-shadow: 0 0 28px rgba(99,102,241,0.35), 0 0 70px rgba(99,102,241,0.12); }
    50%       { box-shadow: 0 0 42px rgba(99,102,241,0.55), 0 0 90px rgba(99,102,241,0.22); }
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 45%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.8px;
    line-height: 1.08;
    margin-bottom: 14px;
}
.hero-subtitle {
    color: #64748b;
    font-size: 15px;
    font-weight: 400;
    letter-spacing: 0.1px;
    margin-bottom: 24px;
}
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    color: #86efac;
    font-size: 12.5px;
    font-weight: 600;
    padding: 6px 18px;
    border-radius: 100px;
    letter-spacing: 0.2px;
}
.badge-dot {
    width: 7px;
    height: 7px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 7px #22c55e;
    display: inline-block;
    flex-shrink: 0;
    animation: blinkDot 2.5s ease-in-out infinite;
}
@keyframes blinkDot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
}

/* ══ GLASS CARD ══ */
.card {
    background: linear-gradient(145deg, rgba(15,21,47,0.92), rgba(10,16,36,0.96));
    border: 1px solid rgba(99,102,241,0.14);
    border-radius: 20px;
    padding: 26px 30px;
    box-shadow: 0 8px 42px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.03) inset;
    backdrop-filter: blur(18px);
    margin-bottom: 20px;
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}
.card:hover {
    box-shadow: 0 14px 56px rgba(99,102,241,0.14), 0 1px 0 rgba(255,255,255,0.03) inset;
    border-color: rgba(99,102,241,0.22);
}

/* ══ UPLOAD DROPZONE override ══ */
div[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(145deg, rgba(15,21,47,0.85), rgba(10,16,36,0.9)) !important;
    border: 2px dashed rgba(99,102,241,0.28) !important;
    border-radius: 18px !important;
    padding: 60px 32px !important;
    text-align: center !important;
    cursor: pointer !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}
div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(139,92,246,0.55) !important;
    box-shadow: 0 0 36px rgba(99,102,241,0.1) !important;
}
div[data-testid="stFileUploaderDropzone"] svg {
    stroke: #6366f1 !important;
    width: 48px !important;
    height: 48px !important;
    margin-bottom: 4px !important;
}
div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] p {
    color: #94a3b8 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stFileUploaderDropzone"] small {
    color: #475569 !important;
    font-size: 12px !important;
}
[data-testid="stFileUploaderFileName"] { color: #94a3b8 !important; font-size: 13px !important; }
[data-testid="stFileUploaderFile"] {
    background: rgba(99,102,241,0.07) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
}

/* ══ IMAGE ══ */
div[data-testid="stImage"] img {
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
}
div[data-testid="stImage"] p {
    color: #475569 !important;
    font-size: 11.5px !important;
    text-align: center !important;
    margin-top: 7px !important;
}

/* ══ SECTION HEADING ══ */
.sec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.3px;
    margin-bottom: 20px;
}

/* ══ SEVERITY PILL ══ */
.sev-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 700;
    padding: 8px 20px;
    border-radius: 100px;
    margin-bottom: 16px;
    letter-spacing: 0.1px;
}
.pill-g { background: rgba(34,197,94,0.12);  border: 1.5px solid rgba(34,197,94,0.38);  color: #86efac; }
.pill-y { background: rgba(234,179,8,0.12);  border: 1.5px solid rgba(234,179,8,0.38);  color: #fef08a; }
.pill-o { background: rgba(249,115,22,0.12); border: 1.5px solid rgba(249,115,22,0.38); color: #fdba74; }
.pill-r { background: rgba(239,68,68,0.12);  border: 1.5px solid rgba(239,68,68,0.38);  color: #fca5a5; }

/* ══ ALERTS ══ */
.alert {
    border-radius: 10px;
    padding: 13px 16px;
    font-size: 13.5px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    line-height: 1.55;
}
.a-ok  { background: rgba(34,197,94,0.07);  border-left: 3px solid #22c55e; color: #86efac; }
.a-ye  { background: rgba(251,191,36,0.07); border-left: 3px solid #fbbf24; color: #fcd34d; }
.a-re  { background: rgba(239,68,68,0.07);  border-left: 3px solid #ef4444; color: #fca5a5; }
.a-bl  { background: rgba(96,165,250,0.07); border-left: 3px solid #60a5fa; color: #93c5fd; }

/* ══ CONFIDENCE ══ */
.conf-wrap  { padding-top: 20px; margin-top: 20px; border-top: 1px solid rgba(99,102,241,0.1); }
.conf-row   { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }
.conf-lbl   { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.9px; color: #475569; }
.conf-val   {
    font-size: 32px; font-weight: 800; line-height: 1;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.conf-track { width: 100%; height: 6px; background: rgba(255,255,255,0.05); border-radius: 100px; overflow: hidden; }
.conf-fill  {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #6366f1 0%, #a78bfa 60%, #60a5fa 100%);
    box-shadow: 0 0 14px rgba(99,102,241,0.55);
    transition: width 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

/* ══ PROB TABLE ROWS ══ */
.prob-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 9px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    transition: background 0.2s ease;
}
.prob-row:hover    { background: rgba(99,102,241,0.06); }
.prob-dot  { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.prob-name { font-size: 13px; font-weight: 500; color: #94a3b8; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prob-trk  { flex: 2; height: 6px; background: rgba(255,255,255,0.05); border-radius: 100px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 100px; }
.prob-pct  { font-size: 13px; font-weight: 700; width: 46px; text-align: right; flex-shrink: 0; }

/* ══ CHANGE IMAGE BUTTON ══ */
div[data-testid="stButton"] button {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.28) !important;
    color: #a5b4fc !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.22s ease !important;
    line-height: 1.4 !important;
}
div[data-testid="stButton"] button:hover {
    background: rgba(99,102,241,0.2) !important;
    border-color: rgba(139,92,246,0.5) !important;
    box-shadow: 0 0 18px rgba(99,102,241,0.22) !important;
    transform: translateY(-1px) !important;
}

/* ══ SPINNER ══ */
.stSpinner > div { border-top-color: #818cf8 !important; }

/* ══ FOOTER ══ */
.footer {
    text-align: center;
    color: #2d3a52;
    font-size: 12px;
    padding: 28px 0 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    border-top: 1px solid rgba(99,102,241,0.07);
    margin-top: 16px;
    letter-spacing: 0.1px;
}

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0B0F1A; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.35); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Load model ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_resnet_model("saved_models/dr_model_resnet50.pth")


try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ── Lookup maps ──────────────────────────────────────────────────────────────────
SEVERITY_COLOR = {
    "No Diabetic Retinopathy":            "#22c55e",
    "Mild Diabetic Retinopathy":          "#eab308",
    "Moderate Diabetic Retinopathy":      "#f97316",
    "Severe Diabetic Retinopathy":        "#ef4444",
    "Proliferative Diabetic Retinopathy": "#dc2626",
}
SEVERITY_EMOJI = {
    "No Diabetic Retinopathy":            "🟢",
    "Mild Diabetic Retinopathy":          "🟡",
    "Moderate Diabetic Retinopathy":      "🟠",
    "Severe Diabetic Retinopathy":        "🔴",
    "Proliferative Diabetic Retinopathy": "🔴",
}
PILL_CLS = {
    "No Diabetic Retinopathy":            "pill-g",
    "Mild Diabetic Retinopathy":          "pill-y",
    "Moderate Diabetic Retinopathy":      "pill-o",
    "Severe Diabetic Retinopathy":        "pill-r",
    "Proliferative Diabetic Retinopathy": "pill-r",
}


# ── Hero header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-ring">👁️</div>
    <div class="hero-title">Diabetic Retinopathy<br>Detection</div>
    <div class="hero-subtitle">Upload a retinal fundus image to get AI-powered severity classification</div>
    <span class="model-badge"><span class="badge-dot"></span>ResNet-50 Model</span>
</div>
""", unsafe_allow_html=True)


# ── Upload / Image preview (dynamic) ────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload retinal image",
    type=["jpg", "jpeg", "png"],
    key=f"img_{st.session_state.uploader_key}",
    label_visibility="collapsed",
)

if uploaded_file:
    # After upload → hide the dropzone widget; show image preview + change button
    st.markdown("""
    <style>
        div[data-testid="stFileUploaderDropzone"] { display: none !important; }
        div[data-testid="stFileUploaderFile"]      { display: none !important; }
        div[data-testid="stFileUploader"] label    { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card" style="padding:20px 24px;">', unsafe_allow_html=True)
    col_img, col_btn = st.columns([5, 1])
    with col_img:
        st.image(uploaded_file, caption="Uploaded fundus image", use_container_width=True)
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Change\nImage", use_container_width=True, key="change_btn"):
            st.session_state.uploader_key += 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Inference ────────────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(uploaded_file.name)[-1], delete=False
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("🔬 Analyzing retinal image with ResNet-50..."):
            image_array = preprocess(tmp_path)
            predicted_label, probabilities = predict(model, image_array)

        confidence  = probabilities[predicted_label]
        conf_pct    = confidence * 100
        pill_cls    = PILL_CLS.get(predicted_label, "pill-g")
        emoji       = SEVERITY_EMOJI.get(predicted_label, "⚪")

        # ── DIAGNOSIS CARD ────────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">✨ Diagnosis Result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sev-pill {pill_cls}">{emoji} {predicted_label}</div>',
                    unsafe_allow_html=True)

        if predicted_label == "No Diabetic Retinopathy":
            st.markdown(
                '<div class="alert a-ok">✅ No signs of Diabetic Retinopathy detected. Your retina appears healthy.</div>',
                unsafe_allow_html=True)
        elif predicted_label == "Mild Diabetic Retinopathy":
            st.markdown(f'<div class="alert a-ye">⚠️ {predicted_label} detected.</div>', unsafe_allow_html=True)
            st.markdown('<div class="alert a-bl">🗓️ Early signs present. Schedule a follow-up with an ophthalmologist.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert a-re">🚨 {predicted_label} detected.</div>', unsafe_allow_html=True)
            st.markdown('<div class="alert a-ye">⚠️ Significant retinal damage detected. Please consult an ophthalmologist immediately.</div>',
                        unsafe_allow_html=True)

        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-row">
                <span class="conf-lbl">Confidence Score</span>
                <span class="conf-val">{conf_pct:.1f}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{conf_pct:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── CHARTS SECTION ────────────────────────────────────────────────────────
        st.markdown('<div class="sec-head" style="padding:4px 0 2px;">📊 Class-wise Probabilities</div>',
                    unsafe_allow_html=True)

        labels_full  = DR_LABELS
        labels_short = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        values       = [probabilities.get(lbl, 0.0) for lbl in labels_full]
        colors       = [SEVERITY_COLOR[lbl] for lbl in labels_full]

        col_donut, col_radar = st.columns(2)

        # ── Donut chart ──────────────────────────────────────────────────────────
        with col_donut:
            st.markdown('<div class="card" style="padding:18px 16px;">', unsafe_allow_html=True)
            fig_donut = go.Figure(go.Pie(
                labels=labels_short,
                values=values,
                hole=0.64,
                marker=dict(colors=colors, line=dict(color='#0B0F1A', width=3)),
                textinfo='none',
                sort=False,
                hovertemplate='<b>%{label}</b><br>%{percent:.1%}<extra></extra>',
            ))
            fig_donut.add_annotation(
                text=f"<b>{conf_pct:.1f}%</b><br>Predicted<br>Confidence",
                x=0.5, y=0.5,
                font=dict(size=15, color='#e2e8f0', family='Inter'),
                showarrow=False,
                xanchor='center',
                yanchor='middle',
                align='center',
            )
            fig_donut.update_layout(
                showlegend=True,
                legend=dict(
                    orientation='v',
                    x=1.02, y=0.5,
                    xanchor='left', yanchor='middle',
                    font=dict(size=11, color='#94a3b8', family='Inter'),
                    bgcolor='rgba(0,0,0,0)',
                    itemclick=False,
                    itemsizing='constant',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=12, b=12, l=0, r=88),
                height=245,
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Radar chart ──────────────────────────────────────────────────────────
        with col_radar:
            st.markdown('<div class="card" style="padding:18px 16px;">', unsafe_allow_html=True)
            vals_c = values + [values[0]]
            lbls_c = labels_short + [labels_short[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_c,
                theta=lbls_c,
                fill='toself',
                fillcolor='rgba(99,102,241,0.16)',
                line=dict(color='#818cf8', width=2.5),
                marker=dict(color='#a78bfa', size=5, line=dict(color='#6366f1', width=1.5)),
                hovertemplate='<b>%{theta}</b><br>%{r:.1%}<extra></extra>',
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        color='#1e293b',
                        gridcolor='rgba(71,85,105,0.2)',
                        linecolor='rgba(71,85,105,0.25)',
                        tickfont=dict(size=8, color='#475569'),
                        tickformat='.0%',
                        dtick=0.25,
                        showticklabels=True,
                    ),
                    angularaxis=dict(
                        color='#475569',
                        gridcolor='rgba(71,85,105,0.18)',
                        linecolor='rgba(71,85,105,0.18)',
                        tickfont=dict(size=11, color='#94a3b8', family='Inter'),
                    ),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(t=24, b=24, l=48, r=48),
                height=245,
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Probability table ────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for lbl in labels_full:
            val     = probabilities.get(lbl, 0.0)
            hex_c   = SEVERITY_COLOR[lbl]
            bar_pct = val * 100
            row_bg  = f'rgba(99,102,241,0.07)' if lbl == predicted_label else 'transparent'
            st.markdown(f"""
            <div class="prob-row" style="background:{row_bg};">
                <div class="prob-dot" style="background:{hex_c};box-shadow:0 0 7px {hex_c}88;"></div>
                <span class="prob-name">{lbl}</span>
                <div class="prob-trk">
                    <div class="prob-fill" style="width:{bar_pct:.1f}%;background:{hex_c};box-shadow:0 0 8px {hex_c}66;"></div>
                </div>
                <span class="prob-pct" style="color:{hex_c};">{bar_pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error during inference: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    # ── Empty state ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:14px 0 26px;color:#334155;font-size:13px;letter-spacing:0.1px;">
        Supported formats: JPG, JPEG, PNG &nbsp;·&nbsp; High-quality fundus photographs work best
    </div>
    """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ℹ️ This AI model is intended for screening purposes only and is not a substitute for professional medical diagnosis.
</div>
""", unsafe_allow_html=True)
