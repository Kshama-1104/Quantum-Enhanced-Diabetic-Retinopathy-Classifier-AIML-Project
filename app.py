import os
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Diabetic Retinopathy Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ["Abnormal", "Normal"]

st.markdown("""
<style>
.stApp { background: #f6faf9; }
.block-container { padding-top: 2rem; max-width: 1180px; }
.hero {
  padding: 28px 32px;
  background: linear-gradient(135deg, #12312d, #1f8a78);
  border-radius: 16px;
  color: white;
  margin-bottom: 24px;
}
.hero h1 { margin: 0; font-size: 44px; }
.hero p { color: #d9f5ee; font-size: 18px; }
.result-box {
  padding: 22px;
  border-radius: 14px;
  background: white;
  border: 1px solid #e3ece9;
}
.small-note { color: #697b77; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

def setting(name, default=None):
    return os.environ.get(name) or st.secrets.get(name, default)

def load_one_model(path):
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def load_model():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_filename = setting("MODEL_FILENAME", "")
    folder_id = setting("MODEL_FOLDER_ID", "")
    file_id = setting("MODEL_FILE_ID", "")

    if file_id and model_filename:
        model_path = models_dir / model_filename
        if not model_path.exists():
            gdown.download(id=file_id, output=str(model_path), quiet=False)

    elif folder_id:
        if not list(models_dir.rglob("*.keras")):
            gdown.download_folder(id=folder_id, output=str(models_dir), quiet=False)

    candidates = list(models_dir.rglob(model_filename)) if model_filename else list(models_dir.rglob("*.keras"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)

    if not candidates:
        raise RuntimeError("No .keras model file found. Check your Google Drive folder.")

    errors = []
    for path in candidates:
        try:
            model = load_one_model(path)
            return model, path.name
        except Exception as e:
            errors.append(f"{path.name}: {e}")

    raise RuntimeError("No model file could be loaded.\n\n" + "\n\n".join(errors))

def prepare_image(uploaded_image, size):
    image = Image.open(uploaded_image).convert("RGB").resize(size)
    array = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0), image

st.markdown("""
<div class="hero">
  <h1>Diabetic Retinopathy Classifier</h1>
  <p>Upload a retinal image and get an AI screening result with confidence scores.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("App Panel")
    st.write("Model source: Google Drive")
    threshold = st.slider("High confidence threshold", 50, 95, 75)
    st.info("This tool is for AI screening support only, not medical diagnosis.")

try:
    model, model_name = load_model()
    st.sidebar.success(f"Loaded: {model_name}")

    tab1, tab2, tab3 = st.tabs(["Classify", "Model Info", "User Guide"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload retinal image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        if uploaded:
            input_shape = model.input_shape
            image_size = (input_shape[1] or 299, input_shape[2] or 299)
            batch, preview = prepare_image(uploaded, image_size)

            with st.spinner("Analyzing retinal image"):
                prediction = model.predict(batch, verbose=0)[0]

            if len(prediction.shape) == 0 or prediction.shape[-1] == 1:
                normal_prob = float(np.ravel(prediction)[0])
                probs = np.array([1 - normal_prob, normal_prob])
            else:
                probs = prediction / prediction.sum()

            top = int(np.argmax(probs))
            confidence = float(probs[top] * 100)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(preview, caption="Uploaded retinal image", use_container_width=True)

            with col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("Screening Result")
                st.metric("Prediction", CLASS_NAMES[top])
                st.metric("Confidence", f"{confidence:.1f}%")

                if confidence >= threshold:
                    st.success("High confidence result")
                else:
                    st.warning("Low confidence result. Try a clearer retinal image.")

                st.write("Class probabilities")
                for name, prob in zip(CLASS_NAMES, probs):
                    st.write(f"{name}: {prob * 100:.1f}%")
                    st.progress(float(prob))

                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("Upload a retinal image to begin classification.")

    with tab2:
        st.subheader("Model Information")
        st.write(f"Loaded model: `{model_name}`")
        st.write(f"Input shape: `{model.input_shape}`")
        st.write("Classes: `Abnormal`, `Normal`")
        st.write("Image preprocessing: resized and rescaled to 0-1.")

    with tab3:
        st.subheader("How to Use")
        st.write("1. Upload a clear retinal fundus image.")
        st.write("2. Wait for the AI model to analyze it.")
        st.write("3. Review the prediction and confidence score.")
        st.write("4. Use the result only as screening support.")

except Exception as e:
    st.error("Model could not be loaded")
    st.code(str(e))
