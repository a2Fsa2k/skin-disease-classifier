"""Streamlit app for HAM10000 skin disease prediction."""

from pathlib import Path

import streamlit as st
from PIL import Image

from src.skin_disease.config import AppConfig
from src.skin_disease.inference import Predictor


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


@st.cache_resource
def load_predictor() -> Predictor:
    config = AppConfig.from_json(CONFIG_PATH)
    return Predictor.from_config(config, project_root=PROJECT_ROOT)


def main() -> None:
    st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
    st.title("Skin Disease Classifier")
    st.caption("Mini-project demo using HAM10000 and a CPU-friendly PyTorch model.")

    predictor = load_predictor()
    uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

    st.info(
        "This tool is a classroom demo, not a medical system. "
        "Predictions depend on the quality of the trained checkpoint."
    )

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Running inference on CPU..."):
            result = predictor.predict(image)

        st.subheader(result.label)
        st.metric("Confidence", f"{result.confidence * 100:.2f}%")

        st.write("Top predictions")
        for class_name, score in result.top_predictions:
            st.write(f"- {class_name}: {score * 100:.2f}%")

        if result.warning:
            st.warning(result.warning)


if __name__ == "__main__":
    main()
