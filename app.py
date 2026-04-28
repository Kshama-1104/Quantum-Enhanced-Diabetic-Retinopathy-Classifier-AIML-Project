import os
from pathlib import Path
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="wide")

CLASS_NAMES = ["Abnormal", "Normal"]

def setting(name, default=None):
    return os.environ.get(name) or st.secrets.get(name, default)

@st.cache_resource
def load_model():
    model_file = setting("MODEL_FILENAME", "best_quantum_inception_cpu.keras")
    model_path = Path("models") / model_file
    file_id = setting("MODEL_FILE_ID")

    if not model_path.exists():
        if not file_id:
            raise RuntimeError("MODEL_FILE_ID is missing. Add it in Streamlit secrets.")
        model_path.parent.mkdir(exist_ok=True)
        gdown.download(id=file_id, output=str(model_path), quiet=False)

    return tf.keras.models.load_model(model_path, compile=False)

def prepare_image(uploaded_image, size):
    image = Image.open(uploaded_image).convert("RGB").resize(size)
    array = np.asarray(image).astype("float32")
    return np.expand_dims(array, axis=0), image

st.title("Diabetic Retinopathy Classifier")
st.caption("AI screening support only. This is not a medical diagnosis.")

uploaded = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png", "bmp", "webp"])

try:
    model = load_model()
    st.success("Model loaded successfully")

    if uploaded:
        input_shape = model.input_shape
        image_size = (input_shape[1] or 299, input_shape[2] or 299)

        batch, preview = prepare_image(uploaded, image_size)
        prediction = model.predict(batch, verbose=0)[0]

        if len(prediction.shape) == 0 or prediction.shape[-1] == 1:
            normal_prob = float(np.ravel(prediction)[0])
            probs = [1 - normal_prob, normal_prob]
        else:
            probs = prediction / prediction.sum()

        top = int(np.argmax(probs))

        col1, col2 = st.columns(2)
        with col1:
            st.image(preview, caption="Uploaded image", use_container_width=True)
        with col2:
            st.metric("Prediction", CLASS_NAMES[top])
            st.metric("Confidence", f"{probs[top] * 100:.1f}%")
            for name, prob in zip(CLASS_NAMES, probs):
                st.write(f"{name}: {prob * 100:.1f}%")
                st.progress(float(prob))
    else:
        st.info("Upload an image to classify it.")

except Exception as e:
    st.error("Model could not be loaded")
    st.code(str(e))
