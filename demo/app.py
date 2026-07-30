import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import joblib
from skimage.feature import local_binary_pattern
from pathlib import Path

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Land Cover AI",
    page_icon="🌍",
    layout="wide"
)

# =========================
# Custom CSS & Professional Styling
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f8fafc;
}

.main-title {
    text-align: center;
    color: #0f172a;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #475569;
    font-size: 16px;
    margin-bottom: 30px;
}

.overview-box {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    max-width: 900px;
    margin: 0 auto 35px auto;
    color: #334155;
    font-size: 15px;
    line-height: 1.7;
    text-align: center;
}

.overview-box h3 {
    color: #0f172a;
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 20px;
}

.result-box {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    text-align: center;
    font-size: 18px;
    color: white;
    padding: 18px;
    border-radius: 10px;
    margin-top: 20px;
    box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.2);
}
</style>
""", unsafe_allow_html=True)

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "ann_fusion_model.h5"
SCALER_PATH = BASE_DIR / "model" / "fusion_scaler.pkl"
ENCODER_PATH = BASE_DIR / "model" / "fusion_label_encoder.pkl"
SAMPLE_DIR = BASE_DIR / "samples"

# =========================
# Header
# =========================
st.markdown('<h1 class="main-title">Land Cover AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Satellite Image Classification Platform</p>', unsafe_allow_html=True)

# =========================
# Overview Section
# =========================
st.markdown(
"""
<div class="overview-box">
    <h3>Overview</h3>
    Land Cover AI is an artificial intelligence platform designed to classify satellite images into distinct land cover categories. 
    The system utilizes advanced feature extraction combining color histograms and local texture patterns (LBP) to accurately categorize images. 
    <br><br>
    Upload a satellite image or choose from the gallery examples below to test the classification model.
</div>
""",
unsafe_allow_html=True
)

# =========================
# LBP Parameters & Model Load
# =========================
LBP_POINTS = 16
LBP_RADIUS = 2
LBP_METHOD = "uniform"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_enc = joblib.load(ENCODER_PATH)
    return model, scaler, label_enc

model, scaler, label_enc = load_model()

# =========================
# Feature Extraction Functions
# =========================
def extract_color_feature(img_bgr, bins_per_channel=8):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([img_hsv], [0], None, [bins_per_channel], [0, 180])
    s_hist = cv2.calcHist([img_hsv], [1], None, [bins_per_channel], [0, 256])
    v_hist = cv2.calcHist([img_hsv], [2], None, [bins_per_channel], [0, 256])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    return np.concatenate([h_hist, s_hist, v_hist])

def extract_lbp_feature(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)
    return hist

def predict_fusion_image(image):
    img = cv2.resize(image, (64, 64))
    color_feat = extract_color_feature(img)
    lbp_feat = extract_lbp_feature(img)
    fusion = np.concatenate([color_feat, lbp_feat]).reshape(1, -1)
    fusion_scaled = scaler.transform(fusion)
    probs = model.predict(fusion_scaled, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_class = label_enc.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]
    return pred_class, confidence

# =========================
# Main Layout (Columns)
# =========================
col1, col2 = st.columns(2, gap="large")

# --- Column 1: Upload Section ---
with col1:
    with st.container(border=True):
        st.subheader("Upload Satellite Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if st.button("Predict Uploaded Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    pred, conf = predict_fusion_image(img)
                st.markdown(f"""
                <div class="result-box">
                <b>Predicted Class:</b> {pred}<br>
                <span style="font-size:15px; color:#e0f2fe;">Confidence: {conf:.2%}</span>
                </div>
                """, unsafe_allow_html=True)

# --- Column 2: Gallery Section ---
with col2:
    with st.container(border=True):
        st.subheader("Satellite Image Gallery")
        samples = {
            "Forest": "forest.jpg",
            "River": "river.jpg",
            "Sea / Lake": "seaLake.jpg"
        }
        selected_sample = st.selectbox("Choose a sample image:", list(samples.keys()))
        sample_path = SAMPLE_DIR / samples[selected_sample]
        
        sample_img = cv2.imread(str(sample_path))
        if sample_img is None:
            st.error(f"Cannot load image: {sample_path}")
        else:
            st.image(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB), caption=selected_sample, use_container_width=True)
            
            if st.button("Predict Sample Image", use_container_width=True):
                with st.spinner("Analyzing sample..."):
                    pred, conf = predict_fusion_image(sample_img)
                st.markdown(f"""
                <div class="result-box">
                <b>Predicted Class:</b> {pred}<br>
                <span style="font-size:15px; color:#e0f2fe;">Confidence: {conf:.2%}</span>
                </div>
                """, unsafe_allow_html=True)