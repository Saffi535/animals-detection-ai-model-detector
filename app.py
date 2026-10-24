import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="WildGuard - Animal Identifier",
    page_icon="🐾",
    layout="centered"
)

# 2. Pyara Dynamic Theme & Animal Background CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.75)), 
                    url('https://images.unsplash.com/photo-1534177616072-ef7dc120449d?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }
    .main-box {
        background: rgba(255, 255, 255, 0.12);
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 20px;
    }
    .custom-title {
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        color: #00ffcc;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
    }
    .custom-subtitle {
        text-align: center;
        font-size: 16px;
        color: #e0e0e0;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading (Cached taake baar baar load na ho)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("animal_detector_model.h5")

model = load_model()

# Model ke exact classes according to training indices
CLASSES = ['Cat', 'Elephant', 'Lion', 'Panda', 'Tiger']
CONFIDENCE_THRESHOLD = 0.75  # 75% se kam ho toh "Not an Animal" trigger hoga

# 4. Header UI
st.markdown("<div class='custom-title'>🐾 WildGuard AI</div>", unsafe_allow_html=True)
st.markdown("<div class='custom-subtitle'>Smart Animal Detector (Cat, Elephant, Lion, Panda, Tiger)</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an animal photo...", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing image features..."):
            # Image preprocessing (224x224 match with MobileNetV2)
            img = image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Model prediction
            predictions = model.predict(img_array)[0]
            max_index = np.argmax(predictions)
            max_confidence = predictions[max_index]
            predicted_animal = CLASSES[max_index]

            st.write("### Detection Result:")
            
            # Non-animal ya Unknown detection check
            if max_confidence < CONFIDENCE_THRESHOLD:
                st.error("🚫 **Error: This is not an animal!** (or animal not supported in model)")
                st.caption(f"Top guess was {predicted_animal} with only {max_confidence*100:.1f}% confidence.")
            else:
                st.success(f"🎉 **Detected Animal:** {predicted_animal}")
                st.info(f"📊 **Confidence Score:** {max_confidence * 100:.2f}%")
                st.progress(float(max_confidence))