from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from helpers import training_utils

IMG_SIZE = (224, 224)
SAVED_MODELS_DIR = Path("saved_models")


def discover_models(saved_models_dir: Path) -> dict[str, dict]:
    """Discover models from metadata files in saved_models/."""
    models: dict[str, dict] = {}
    if not saved_models_dir.exists():
        return models

    for meta_path in sorted(saved_models_dir.glob("*_meta.json")):
        model_name = meta_path.name.replace("_meta.json", "")
        model_path = saved_models_dir / f"{model_name}.keras"
        if not model_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            continue

        threshold = float(meta.get("threshold", 0.5))
        metrics = meta.get("metrics", {})
        models[model_name] = {
            "model_path": model_path,
            "meta_path": meta_path,
            "threshold": threshold,
            "metrics": metrics,
            "meta": meta,
        }
    return models


@st.cache_resource(show_spinner=False)
def load_selected_model(model_path: str):
    """Load and cache a Keras model by path."""
    return training_utils.load_model_compat(model_path, compile=False)


def preprocess_upload(uploaded_file, img_size: tuple[int, int]) -> tuple[Image.Image, np.ndarray]:
    """Convert uploaded image to display image + model-ready batch."""
    pil_img = Image.open(uploaded_file).convert("RGB")
    model_img = pil_img.resize(img_size)
    x = np.asarray(model_img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return pil_img, x


def render_metrics(metrics: dict):
    """Show model metrics in sidebar."""
    if not metrics:
        st.sidebar.caption("No metrics found in metadata.")
        return

    cols = st.sidebar.columns(2)
    keys = ["accuracy", "auc", "f1", "precision", "recall"]
    labels = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
    }

    for i, key in enumerate(keys):
        if key in metrics:
            cols[i % 2].metric(labels[key], f"{float(metrics[key]):.4f}")


def main():
    st.set_page_config(page_title="Pneumonia Inference", page_icon="🩺", layout="wide")
    st.title("Pneumonia Detection Inference")
    st.caption("Upload a chest X-ray, pick a saved model, and compare predictions.")

    models = discover_models(SAVED_MODELS_DIR)
    if not models:
        st.error(
            "No saved models found. Expected files in `saved_models/`:\n"
            "- `<model_name>.keras`\n"
            "- `<model_name>_meta.json`"
        )
        st.info("Run your notebook save cells first, then restart this app.")
        return

    st.sidebar.header("Model Selection")
    selected_name = st.sidebar.selectbox("Choose model", list(models.keys()))
    selected = models[selected_name]
    threshold = selected["threshold"]

    st.sidebar.markdown("### Selected Model Info")
    st.sidebar.write(f"**Name:** `{selected_name}`")
    st.sidebar.write(f"**Threshold:** `{threshold:.3f}`")
    render_metrics(selected["metrics"])

    uploaded_file = st.file_uploader(
        "Upload chest X-ray image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload an image to run inference.")
        return

    display_img, batch = preprocess_upload(uploaded_file, IMG_SIZE)
    model = load_selected_model(str(selected["model_path"]))

    prob = float(model(batch, training=False).numpy().squeeze())
    prob = float(np.clip(prob, 0.0, 1.0))
    pred_label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    confidence = prob if pred_label == "PNEUMONIA" else (1.0 - prob)

    left, right = st.columns([1, 1])
    with left:
        st.image(display_img, caption="Uploaded X-ray", use_container_width=True)
    with right:
        if pred_label == "PNEUMONIA":
            st.error(f"Prediction: **{pred_label}**")
        else:
            st.success(f"Prediction: **{pred_label}**")

        st.metric("P(PNEUMONIA)", f"{prob:.4f}")
        st.metric("Decision threshold", f"{threshold:.4f}")
        st.metric("Confidence", f"{confidence:.2%}")
        st.progress(prob)

        st.caption(
            "Inference uses raw resized image pixels; model-internal preprocessing "
            "(Rescaling for baseline CNN / VGG16 preprocess_input for transfer models) is applied automatically."
        )


if __name__ == "__main__":
    main()
