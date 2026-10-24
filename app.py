import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="WildGuard - Smart Animal Identifier",
    page_icon="🐾",
    layout="wide"
)

# 2. Sidebar - Themes & Settings
st.sidebar.title("🌿 WildGuard Controls")

# High-reliability Direct Background Images
backgrounds = {
    "Safari Lion": "https://images.pexels.com/photos/247502/pexels-photo-247502.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "Deep Jungle": "https://images.pexels.com/photos/142497/pexels-photo-142497.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "Tiger Sanctuary": "https://images.pexels.com/photos/145939/pexels-photo-145939.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "Elephant Valley": "https://images.pexels.com/photos/66898/elephant-cub-tsavo-kenya-66898.jpeg?auto=compress&cs=tinysrgb&w=1920"
}

selected_bg = st.sidebar.selectbox("Choose Background Theme:", list(backgrounds.keys()))
bg_url = backgrounds[selected_bg]

confidence_cutoff = st.sidebar.slider(
    "Animal Detection Sensitivity (%)", 
    min_value=50, 
    max_value=95, 
    value=75, 
    step=5
) / 100.0

# 3. Enhanced CSS (Fixes contrast, radio labels, and transparent glass container)
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(10, 15, 25, 0.72), rgba(10, 15, 25, 0.85)), 
                    url('{bg_url}') no-repeat center center fixed !important;
        background-size: cover !important;
    }}
    
    /* Ensure all text & radio button labels stay bright & crisp */
    h1, h2, h3, h4, p, label, .stMarkdown, .stRadio label span {{
        color: #f8fafc !important;
        font-weight: 500 !important;
    }}
    
    .block-container {{
        background: rgba(15, 23, 42, 0.65);
        border-radius: 20px;
        padding: 30px !important;
        margin-top: 20px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }}
    
    .metric-card {{
        background: rgba(255, 255, 255, 0.08);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-top: 15px;
    }}
    </style>
""", unsafe_allow_html=True)

# 4. Animal Facts Dictionary
ANIMAL_INFO = {
    "Cat": {"diet": "Carnivore", "lifespan": "12-18 years", "fact": "Cats can rotate their ears 180 degrees and spend 70% of their lives sleeping."},
    "Elephant": {"diet": "Herbivore", "lifespan": "60-70 years", "fact": "Elephants are the world's largest land animals and communicate via low-frequency seismic waves."},
    "Lion": {"diet": "Carnivore", "lifespan": "10-14 years", "fact": "A lion's roar can be heard from up to 8 kilometers (5 miles) away."},
    "Panda": {"diet": "Herbivore (Bamboo)", "lifespan": "20 years", "fact": "Pandas spend around 12 hours a day eating bamboo to satisfy their daily energy needs."},
    "Tiger": {"diet": "Carnivore", "lifespan": "10-15 years", "fact": "No two tigers have identical stripes; their patterns are unique like human fingerprints."}
}

# 5. Load Trained Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("animal_detector_model.h5")

model = load_model()
CLASSES = ['Cat', 'Elephant', 'Lion', 'Panda', 'Tiger']

# 6. Main UI
st.title("🐾 WildGuard AI: Animal Identifier")
st.write("Detect cats, elephants, lions, pandas, and tigers — or flag non-animal objects.")

input_method = st.radio("Choose Input Mode:", ["📁 File Uploader", "📸 Live Camera Snapshot"], horizontal=True)

image = None
if input_method == "📁 File Uploader":
    uploaded_file = st.file_uploader("Upload an animal image...", type=["jpg", "png", "jpeg", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
else:
    camera_file = st.camera_input("Take a photo...")
    if camera_file is not None:
        image = Image.open(camera_file).convert('RGB')

# 7. Detection Pipeline
if image is not None:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.image(image, caption="Current Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)[0]
            max_idx = np.argmax(predictions)
            max_confidence = predictions[max_idx]
            predicted_label = CLASSES[max_idx]

            st.subheader("Analysis Results")
            
            if max_confidence < confidence_cutoff:
                st.error("🚫 **Alert: This is NOT an animal!** (or not supported in model)")
                st.caption(f"Closest match was **{predicted_label}** with only **{max_confidence * 100:.1f}%** confidence.")
            else:
                st.success(f"🎉 **Identified Animal: {predicted_label}**")
                st.metric(label="Prediction Confidence", value=f"{max_confidence * 100:.2f}%")
                
                info = ANIMAL_INFO[predicted_label]
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📖 Quick Animal Facts</h4>
                    <p><b>Diet:</b> {info['diet']}</p>
                    <p><b>Lifespan:</b> {info['lifespan']}</p>
                    <p><b>Did you know?</b> {info['fact']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("---")
            st.write("#### 📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Animal": CLASSES,
                "Probability (%)": [round(float(p) * 100, 2) for p in predictions]
            }).set_index("Animal")
            st.bar_chart(prob_df)
