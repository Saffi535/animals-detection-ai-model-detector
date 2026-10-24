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

# Dynamic Animal Backgrounds
backgrounds = {
    "Deep Jungle": "https://images.unsplash.com/photo-1511497584788-87676104235f?auto=format&fit=crop&w=1920&q=80",
    "Safari Lion": "https://images.unsplash.com/photo-1614027164847-1b28caa1228f?auto=format&fit=crop&w=1920&q=80",
    "Tiger Sanctuary": "https://images.unsplash.com/photo-1534177616072-ef7dc120449d?auto=format&fit=crop&w=1920&q=80",
    "Elephant Valley": "https://images.unsplash.com/photo-1557050543-4d5f4e07ef46?auto=format&fit=crop&w=1920&q=80"
}
selected_bg = st.sidebar.selectbox("Choose Background Theme:", list(backgrounds.keys()))
bg_url = backgrounds[selected_bg]

# Sensitivity Slider for Non-Animal detection
confidence_cutoff = st.sidebar.slider(
    "Animal Detection Sensitivity (%)", 
    min_value=50, 
    max_value=95, 
    value=75, 
    step=5
) / 100.0

# 3. Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.70), rgba(0, 0, 0, 0.80)), 
                    url('{bg_url}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }}
    .metric-card {{
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 14px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    </style>
""", unsafe_allow_html=True)

# 4. Animal Facts Dictionary
ANIMAL_INFO = {
    "Cat": {"diet": "Carnivore", "lifespan": "12-18 years", "fact": "Cats can rotate their ears 180 degrees and spend 70% of their lives sleeping."},
    "Elephant": {"diet": "Herbivore", "lifespan": "60-70 years", "fact": "Elephants are the world's largest land animals and have incredible memory capabilities."},
    "Lion": {"diet": "Carnivore", "lifespan": "10-14 years", "fact": "A lion's roar can be heard from up to 8 kilometers (5 miles) away."},
    "Panda": {"diet": "Herbivore (Bamboo)", "lifespan": "20 years", "fact": "Pandas spend around 12 hours a day eating bamboo to satisfy their daily dietary needs."},
    "Tiger": {"diet": "Carnivore", "lifespan": "10-15 years", "fact": "No two tigers have the same stripes; their stripe patterns are unique like human fingerprints."}
}

# 5. Load Trained Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("animal_detector_model.h5")

model = load_model()
CLASSES = ['Cat', 'Elephant', 'Lion', 'Panda', 'Tiger']

# 6. Main UI
st.title("🐾 WildGuard AI: Animal Identifier")
st.write("Detect cats, elephants, lions, pandas, and tigers — or identify non-animal objects.")

# Input Mode: File Upload OR Live Camera
input_method = st.radio("Choose Input Mode:", ["📁 File Uploader", "📸 Live Camera Snapshot"], horizontal=True)

image = None
if input_method == "📁 File Uploader":
    uploaded_file = st.file_uploader("Upload an animal image...", type=["jpg", "png", "jpeg", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
else:
    camera_file = st.camera_input("Take a photo of an animal...")
    if camera_file is not None:
        image = Image.open(camera_file).convert('RGB')

# 7. Detection & Insights
if image is not None:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.image(image, caption="Current Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing with MobileNetV2..."):
            # Preprocessing
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(img_array)[0]
            max_idx = np.argmax(predictions)
            max_confidence = predictions[max_idx]
            predicted_label = CLASSES[max_idx]

            st.subheader("Analysis Results")
            
            # Non-Animal Threshold Condition
            if max_confidence < confidence_cutoff:
                st.error("🚫 **Alert: This is NOT an animal!** (or animal is not in our trained database)")
                st.warning(f"Closest match was **{predicted_label}** with only **{max_confidence * 100:.1f}%** confidence.")
            else:
                st.success(f"🎉 **Identified Animal: {predicted_label}**")
                st.metric(label="Prediction Confidence", value=f"{max_confidence * 100:.2f}%")
                
                # Animal Info Card
                info = ANIMAL_INFO[predicted_label]
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📖 Quick Animal Facts</h4>
                    <p><b>Diet:</b> {info['diet']}</p>
                    <p><b>Lifespan:</b> {info['lifespan']}</p>
                    <p><b>Did you know?</b> {info['fact']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Probabilities Bar Chart
            st.write("---")
            st.write("#### 📊 Prediction Probabilities for All Classes")
            prob_df = pd.DataFrame({
                "Animal": CLASSES,
                "Probability (%)": [round(float(p) * 100, 2) for p in predictions]
            }).set_index("Animal")
            st.bar_chart(prob_df)
