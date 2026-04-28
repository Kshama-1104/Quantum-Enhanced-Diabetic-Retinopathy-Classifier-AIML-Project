import json
import os
from datetime import datetime
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
  padding: 32px 36px;
  background: linear-gradient(135deg, #12312d, #218a77);
  border-radius: 16px;
  color: white;
  margin-bottom: 24px;
}
.hero h1 { margin: 0; font-size: 46px; line-height: 1.08; }
.hero p { color: #d9f5ee; font-size: 18px; margin-top: 14px; }
.panel {
  padding: 20px;
  border-radius: 12px;
  background: white;
  border: 1px solid #e2ebe8;
}
.result-normal {
  padding: 18px;
  border-radius: 12px;
  background: #e7f7ef;
  color: #075c3c;
  border: 1px solid #bde8d1;
}
.result-abnormal {
  padding: 18px;
  border-radius: 12px;
  background: #fdecec;
  color: #9f1d1d;
  border: 1px solid #f6c7c7;
}
.small-note { color: #61736f; font-size: 14px; }
.footer {
  color:#667;
  font-size:14px;
  border-top: 1px solid #d9e5e1;
  padding-top: 22px;
  margin-top: 48px;
}
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

def predict(model, batch):
    raw = model.predict(batch, verbose=0)[0]

    if len(raw.shape) == 0 or raw.shape[-1] == 1:
        normal_prob = float(np.ravel(raw)[0])
        probs = np.array([1 - normal_prob, normal_prob], dtype="float32")
    else:
        probs = raw.astype("float32")
        probs = probs / max(float(probs.sum()), 1e-8)

    top = int(np.argmax(probs))
    return top, probs

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam(model, batch, class_index):
    layer_name = find_last_conv_layer(model)
    if not layer_name:
        raise RuntimeError("Grad-CAM unavailable: no Conv2D layer found.")

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        if predictions.shape[-1] == 1:
            target = predictions[:, 0] if class_index == 1 else 1 - predictions[:, 0]
        else:
            target = predictions[:, class_index]

    grads = tape.gradient(target, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap, opacity):
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
    heat = np.asarray(heat).astype("float32") / 255.0

    heat_rgb = np.zeros((heat.shape[0], heat.shape[1], 3), dtype="float32")
    heat_rgb[..., 0] = 255 * heat
    heat_rgb[..., 1] = 180 * heat
    heat_rgb[..., 2] = 40 * (1 - heat)

    base = np.asarray(image).astype("float32")
    overlay = base * (1 - opacity) + heat_rgb * opacity
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))

st.markdown("""
<div class="hero">
  <h1>Diabetic Retinopathy Classifier</h1>
  <p>Upload a retinal fundus image and receive an AI-assisted screening prediction with confidence scores and visual explanation.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Screening Settings")
    st.write("Adjust screening sensitivity")
    threshold = st.slider("Confidence threshold", 50, 95, 75)
    show_heatmap = st.toggle("Show Grad-CAM explanation", value=True)
    heatmap_opacity = st.slider("Heatmap intensity", 20, 80, 45) / 100
    st.info("Results should be reviewed by a qualified medical professional.")

try:
    model, model_name = load_model()
    st.sidebar.success("AI model is ready")

    tab1, tab2, tab3, tab4 = st.tabs(["Classify", "Explainability", "Model Info", "Project Guide"])

    uploaded = None
    prediction_data = None

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
                top, probs = predict(model, batch)

            confidence = float(probs[top] * 100)
            label = CLASS_NAMES[top]
            prediction_data = {
                "prediction": label,
                "confidence": round(confidence, 2),
                "abnormal_probability": round(float(probs[0] * 100), 2),
                "normal_probability": round(float(probs[1] * 100), 2),
                "model": model_name,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(preview, caption="Uploaded retinal image", use_container_width=True)

            with col2:
                box_class = "result-abnormal" if label == "Abnormal" else "result-normal"
                st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
                st.subheader("Screening Result")
                st.metric("Prediction", label)
                st.metric("Confidence", f"{confidence:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)

                if confidence >= threshold:
                    st.success("High confidence result")
                else:
                    st.warning("Low confidence result. Try a clearer retinal image.")

                if label == "Abnormal":
                    st.error("Recommendation: Consult an eye-care professional for clinical evaluation.")
                else:
                    st.info("Recommendation: Continue routine screening as advised by a healthcare professional.")

                st.write("Class probabilities")
                for name, prob in zip(CLASS_NAMES, probs):
                    st.write(f"{name}: {prob * 100:.1f}%")
                    st.progress(float(prob))

                st.download_button(
                    "Download result summary",
                    data=json.dumps(prediction_data, indent=2),
                    file_name="screening_result.json",
                    mime="application/json",
                )
        else:
            st.info("Upload a retinal image to begin classification.")

    with tab2:
        st.subheader("Visual Explanation")
        if uploaded and prediction_data:
            try:
                input_shape = model.input_shape
                image_size = (input_shape[1] or 299, input_shape[2] or 299)
                batch, preview = prepare_image(uploaded, image_size)
                top, _ = predict(model, batch)

                if show_heatmap:
                    heatmap = make_gradcam(model, batch, top)
                    overlay = overlay_heatmap(preview, heatmap, heatmap_opacity)
                    st.image(overlay, caption="Grad-CAM heatmap overlay", use_container_width=True)
                    st.caption("Warmer areas contributed more strongly to the model prediction.")
                else:
                    st.info("Enable Grad-CAM explanation from the sidebar.")
            except Exception as e:
                st.warning("Grad-CAM explanation is unavailable for this model.")
                st.code(str(e))
        else:
            st.info("Upload and classify an image first to view the explanation.")

    with tab3:
        st.subheader("Model Information")
        st.write(f"Loaded model: `{model_name}`")
        st.write(f"Input shape: `{model.input_shape}`")
        st.write("Classes: `Abnormal`, `Normal`")
        st.write("Image preprocessing: resized and normalized before prediction.")
        st.write("Deployment: Streamlit Cloud")
        st.write("Model storage: Google Drive")

    with tab4:
        st.subheader("Project Workflow")
        st.write("1. Collect and organize retinal image dataset.")
        st.write("2. Preprocess images and split into training/validation sets.")
        st.write("3. Train transfer learning models using InceptionV3 and ResNet-based experiments.")
        st.write("4. Evaluate models using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.")
        st.write("5. Deploy the best working model through this Streamlit web application.")
        st.write("6. Provide Grad-CAM visual explanation for interpretability.")

except Exception as e:
    st.error("Model could not be loaded")
    st.code(str(e))

st.markdown("""
<div class="footer">
This application provides AI-assisted screening support for diabetic retinopathy.
It is not a substitute for professional diagnosis, clinical examination, or medical advice.
Uploaded images are processed for prediction and are not intentionally stored by this app.
</div>
""", unsafe_allow_html=True)
