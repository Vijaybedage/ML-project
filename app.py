"""
Animal Classification - Streamlit Web App
Author: Vijay Bedage
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import plotly.graph_objects as go
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐾 Animal Classifier",
    page_icon="🐾",
    layout="wide"
)

CLASSES = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
           'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
           'Panda', 'Tiger', 'Zebra']

ANIMAL_EMOJIS = {
    'Bear': '🐻', 'Bird': '🐦', 'Cat': '🐱', 'Cow': '🐄',
    'Deer': '🦌', 'Dog': '🐶', 'Dolphin': '🐬', 'Elephant': '🐘',
    'Giraffe': '🦒', 'Horse': '🐴', 'Kangaroo': '🦘', 'Lion': '🦁',
    'Panda': '🐼', 'Tiger': '🐯', 'Zebra': '🦓'
}

ANIMAL_FACTS = {
    'Bear': 'Bears can smell food from 20 miles away!',
    'Bird': 'There are over 10,000 known species of birds.',
    'Cat': 'Cats sleep 12–16 hours a day.',
    'Cow': 'Cows have almost 360° panoramic vision.',
    'Deer': 'Deer can run up to 30 mph.',
    'Dog': 'Dogs have 300 million olfactory receptors in their nose.',
    'Dolphin': 'Dolphins sleep with one eye open.',
    'Elephant': 'Elephants are the only animals that can\'t jump.',
    'Giraffe': 'A giraffe\'s tongue is 18–20 inches long.',
    'Horse': 'Horses can sleep both standing up and lying down.',
    'Kangaroo': 'Kangaroos can\'t walk backwards.',
    'Lion': 'A lion\'s roar can be heard from 5 miles away.',
    'Panda': 'Giant pandas eat up to 38 pounds of bamboo a day.',
    'Tiger': 'No two tigers have the same stripe pattern.',
    'Zebra': 'Zebra stripes act as a natural bug repellent.'
}

IMG_SIZE = (224, 224)
MODEL_PATH = 'models/animal_classifier.h5'


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Model not found. Please train the model first using `python src/train.py`")
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict(model, img: Image.Image):
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    return probs, top5_idx


# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .big-title { font-size: 2.5rem; font-weight: 800; text-align: center; 
                 background: linear-gradient(90deg, #667eea, #764ba2);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .result-box { background: #f0f9ff; border-left: 4px solid #667eea;
                  padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .fact-box   { background: #fff7ed; border-left: 4px solid #f59e0b;
                  padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🐾 Animal Classification AI</p>', unsafe_allow_html=True)
st.markdown("**Upload an animal image and let the AI identify it!**")
st.markdown("*Built with MobileNetV2 Transfer Learning | 15 Animal Classes | ~95%+ Accuracy*")

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
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file:
        model = load_model()
        if model:
            with st.spinner("🔍 Analyzing image..."):
                probs, top5_idx = predict(model, img)

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
