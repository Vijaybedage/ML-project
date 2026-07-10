"""
Animal Classification - Streamlit Web App
Author: Vijay Bedage
Run: streamlit run app.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import plotly.graph_objects as go

from src.config import (
    CLASSES, ANIMAL_EMOJIS, ANIMAL_FACTS,
    IMG_SIZE, MODEL_PATH
)
from src.validator import validate_prediction, format_rejection_message

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="wide"
)


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model with caching for performance."""
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please train the model first using `python src/train.py`")
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# ─── PREDICT ──────────────────────────────────────────────────────────────────
_FEATURE_EXTRACTOR = None


def _get_feature_extractor():
    """Lazily build and cache the feature extractor for OOD detection."""
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        try:
            from src.ood_detector import build_feature_extractor
            _FEATURE_EXTRACTOR = build_feature_extractor()
        except Exception as e:
            st.warning(f"OOD detection unavailable: {e}")
    return _FEATURE_EXTRACTOR


def predict(model, img: Image.Image):
    """Run inference on a PIL image and return probabilities with validation."""
    try:
        img_resized = img.resize(IMG_SIZE)
        arr = np.array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        probs = model.predict(arr, verbose=0)[0]
        top5_idx = np.argsort(probs)[::-1][:5]
        
        # Try to get features for OOD detection
        features = None
        extractor = _get_feature_extractor()
        if extractor is not None:
            try:
                features = extractor.predict(arr, verbose=0)[0]
            except Exception:
                pass
        
        validation = validate_prediction(probs, image_arr=arr, image_features=features)
        return probs, top5_idx, validation
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None


# ─── UI STYLES ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .big-title { font-size: 2.5rem; font-weight: 800; text-align: center; 
                 background: linear-gradient(90deg, #667eea, #764ba2);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .result-box { background: #f0f9ff; border-left: 4px solid #667eea;
                  padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .fact-box   { background: #fff7ed; border-left: 4px solid #f59e0b;
                  padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .reject-box { background: #fef2f2; border-left: 4px solid #e74c3c;
                  padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🐾 Animal Classification AI</p>', unsafe_allow_html=True)
st.markdown("**Upload an animal image and let the AI identify it!**")
st.markdown("*Built with MobileNetV2 Transfer Learning | 15 Animal Classes | ~93% Validation Accuracy*")

st.divider()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an animal image...",
        type=['jpg', 'jpeg', 'png'],
        help="Supports JPG, JPEG, PNG formats"
    )

    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Could not open image: {str(e)}")
            img = None

with col2:
    if uploaded_file and img is not None:
        model = load_model()
        if model:
            with st.spinner("Analyzing image..."):
                probs, top5_idx, validation = predict(model, img)

            if probs is not None and validation is not None:
                if not validation['is_valid']:
                    rejection_msg = format_rejection_message(validation)
                    st.markdown(f"""
                    <div class="reject-box">
                        <pre style="margin:0; white-space:pre-wrap; font-family:inherit;">{rejection_msg}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    predicted_class = CLASSES[top5_idx[0]]
                    confidence = probs[top5_idx[0]]
                    emoji = ANIMAL_EMOJIS[predicted_class]

                    conf_color = "green" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"

                    st.markdown(f"""
                    <div class="result-box">
                        <h2 style="margin:0">{emoji} {predicted_class}</h2>
                        <p style="color:{conf_color}; font-size:1.2rem; margin:0.5rem 0">
                            Confidence: <strong>{confidence*100:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="fact-box">
                        <strong>🧠 Fun Fact:</strong> {ANIMAL_FACTS[predicted_class]}
                    </div>
                    """, unsafe_allow_html=True)

                    # Validation details (expandable)
                    with st.expander("🔍 Validation Details"):
                        st.write(f"**Top-1:** {validation['top1_class']} ({validation['confidence']*100:.1f}%)")
                        st.write(f"**Top-2:** {validation['top2_class']} ({validation.get('top2_prob', 0)*100:.1f}%)")
                        st.write(f"**Margin:** {validation['margin']*100:.1f}%")
                        st.write(f"**Entropy:** {validation['entropy']:.2f} bits")
                        if validation['ood_distance'] is not None:
                            st.write(f"**OOD Distance:** {validation['ood_distance']:.1f}")

                    # Plotly bar chart
                    st.subheader("📊 Top 5 Predictions")
                    top5_classes = [f"{ANIMAL_EMOJIS[CLASSES[i]]} {CLASSES[i]}" for i in top5_idx]
                    top5_confs = [float(probs[i]) * 100 for i in top5_idx]

                    fig = go.Figure(go.Bar(
                        x=top5_confs,
                        y=top5_classes,
                        orientation='h',
                        marker=dict(
                            color=top5_confs,
                            colorscale='Blues',
                            showscale=False
                        ),
                        text=[f"{c:.1f}%" for c in top5_confs],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        xaxis_title="Confidence (%)",
                        xaxis=dict(range=[0, 110]),
                        height=300,
                        margin=dict(l=20, r=60, t=20, b=40),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Please upload an animal image to get started.")
        st.markdown("#### 🐾 Supported Animals:")
        cols = st.columns(3)
        for i, animal in enumerate(CLASSES):
            cols[i % 3].write(f"{ANIMAL_EMOJIS[animal]} {animal}")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem">
    Built by <strong>Vijay Bedage</strong> | MCA Student, KLE Technological University<br>
    Transfer Learning with MobileNetV2 | TensorFlow & Keras
</div>
""", unsafe_allow_html=True)