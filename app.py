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
    page_title="Diabetic Retinopathy Screening",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_NAMES = ["Abnormal", "Normal"]

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: #f5faf8; color: #172522; }
.block-container { padding-top: 2rem; max-width: 1120px; }
.hero {
  padding: 34px 38px;
  background: linear-gradient(135deg, #10342e, #1f8a76);
  border-radius: 14px;
  color: white;
  margin-bottom: 24px;
}
.hero h1 { margin: 0; font-size: 44px; line-height: 1.1; }
.hero p { margin-top: 14px; color: #daf7f0; font-size: 18px; max-width: 760px; }
.card {
  padding: 22px;
  border-radius: 10px;
  background: white;
  border: 1px solid #dfe9e6;
}
.step {
  padding: 16px;
  border-radius: 10px;
  background: #ffffff;
  border: 1px solid #dfe9e6;
  min-height: 92px;
}
.result-normal {
  padding: 22px;
  border-radius: 10px;
  background: #e7f7ef;
  color: #075c3c;
  border: 1px solid #bde8d1;
}
.result-abnormal {
  padding: 22px;
  border-radius: 10px;
  background: #fdecec;
  color: #9f1d1d;
  border: 1px solid #f3c5c5;
}
.footer-note {
  color: #66736f;
  font-size: 14px;
  border-top: 1px solid #d9e5e1;
  padding-top: 20px;
  margin-top: 42px;
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
        raise RuntimeError("No .keras model file found. Check MODEL_FOLDER_ID or MODEL_FILE_ID.")

    errors = []
    for path in candidates:
        try:
            return load_one_model(path), path.name
        except Exception as e:
            errors.append(f"{path.name}: {e}")

    raise RuntimeError("No model file could be loaded.\n\n" + "\n\n".join(errors))

def prepare_image(uploaded_file, size):
    image = Image.open(uploaded_file).convert("RGB").resize(size)
    array = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0), image

def predict(model, batch):
    raw = model.predict(batch, verbose=0)[0]

    if len(raw.shape) == 0 or raw.shape[-1] == 1:
        normal_probability = float(np.ravel(raw)[0])
        probs = np.array([1 - normal_probability, normal_probability], dtype="float32")
    else:
        probs = raw.astype("float32")
        probs = probs / max(float(probs.sum()), 1e-8)

    top_index = int(np.argmax(probs))
    return top_index, probs

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam(model, batch, class_index):
    layer_name = find_last_conv_layer(model)
    if not layer_name:
        raise RuntimeError("No convolution layer found for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output],
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

def overlay_heatmap(image, heatmap, opacity=0.45):
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
    heat = np.asarray(heatmap_image).astype("float32") / 255.0

    color = np.zeros((heat.shape[0], heat.shape[1], 3), dtype="float32")
    color[..., 0] = 255 * heat
    color[..., 1] = 160 * heat
    color[..., 2] = 30 * (1 - heat)

    base = np.asarray(image).astype("float32")
    mixed = base * (1 - opacity) + color * opacity
    return Image.fromarray(np.uint8(np.clip(mixed, 0, 255)))

st.markdown("""
<div class="hero">
  <h1>Diabetic Retinopathy Screening</h1>
  <p>Upload a retinal fundus image to receive an AI-assisted screening result, confidence score, and visual explanation.</p>
</div>
""", unsafe_allow_html=True)

step1, step2, step3 = st.columns(3)
with step1:
    st.markdown('<div class="step"><b>1. Upload</b><br>Choose a clear retinal image.</div>', unsafe_allow_html=True)
with step2:
    st.markdown('<div class="step"><b>2. Analyze</b><br>The AI model checks visual patterns.</div>', unsafe_allow_html=True)
with step3:
    st.markdown('<div class="step"><b>3. Review</b><br>See result, confidence, and heatmap.</div>', unsafe_allow_html=True)

st.write("")

try:
    model, model_name = load_model()

    uploaded = st.file_uploader(
        "Upload retinal image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if not uploaded:
        st.info("Upload a retinal image to start screening.")
    else:
        input_shape = model.input_shape
        image_size = (input_shape[1] or 299, input_shape[2] or 299)
        batch, preview = prepare_image(uploaded, image_size)

        with st.spinner("Analyzing image"):
            top, probs = predict(model, batch)

        label = CLASS_NAMES[top]
        confidence = float(probs[top] * 100)

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.image(preview, caption="Uploaded retinal image", use_container_width=True)

        with right:
            result_class = "result-abnormal" if label == "Abnormal" else "result-normal"
            friendly_label = "Possible DR signs detected" if label == "Abnormal" else "No DR signs detected"

            st.markdown(f'<div class="{result_class}">', unsafe_allow_html=True)
            st.subheader("Screening Result")
            st.metric("Result", friendly_label)
            st.metric("Confidence", f"{confidence:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            if label == "Abnormal":
                st.error("Recommended action: Please consult an eye-care professional.")
            else:
                st.success("Recommended action: Continue routine screening as advised.")

            st.write("Probability breakdown")
            for name, prob in zip(CLASS_NAMES, probs):
                st.write(f"{name}: {prob * 100:.1f}%")
                st.progress(float(prob))

            report_text = f"""
DIABETIC RETINOPATHY SCREENING REPORT

Result:
{friendly_label}

Confidence:
{confidence:.1f}%

Probability Breakdown:
Abnormal: {float(probs[0] * 100):.1f}%
Normal: {float(probs[1] * 100):.1f}%

Recommended Action:
{"Please consult an eye-care professional for clinical evaluation." if label == "Abnormal" else "Continue routine screening as advised by a healthcare professional."}

Generated At:
{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

Disclaimer:
This report is generated by an AI-assisted screening tool. It is not a medical diagnosis and should be reviewed by a qualified healthcare professional.
"""

            st.download_button(
                "Download screening report",
                data=report_text.strip(),
                file_name="diabetic_retinopathy_screening_report.txt",
                mime="text/plain",
            )

        with st.expander("Show visual explanation"):
            try:
                heatmap = make_gradcam(model, batch, top)
                overlay = overlay_heatmap(preview, heatmap)
                st.image(overlay, caption="Grad-CAM heatmap", use_container_width=True)
                st.caption("Warmer regions show areas that influenced the AI prediction more strongly.")
            except Exception as e:
                st.warning("Grad-CAM explanation is unavailable for this model.")
                st.code(str(e))

        with st.expander("Image quality tips"):
            st.write("- Use a clear retinal fundus image.")
            st.write("- Avoid blurry, dark, or heavily cropped images.")
            st.write("- Keep the retina centered in the image.")

except Exception as e:
    st.error("Model could not be loaded")
    st.code(str(e))

st.markdown("""
<div class="footer-note">
This application provides AI-assisted screening support only. It is not a substitute for professional medical diagnosis,
clinical examination, or advice from a qualified healthcare provider. Uploaded images are processed for prediction and are
not intentionally stored by this app.
</div>
""", unsafe_allow_html=True)
