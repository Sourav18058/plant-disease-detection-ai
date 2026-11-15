"""PlantCare AI - Streamlit version.

Loads the trained Keras model directly (no Flask / API in between) and keeps
the same look as the Flask design: green header, white card, dashed upload box.
Run with:  streamlit run streamlit_app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "plant_disease_model.keras"
CLASSES_PATH = BASE_DIR / "model" / "class_names.json"

IMG_SIZE = (224, 224)      # model input size (from the saved model config)
SCALE_TO_01 = True         # divide pixels by 255 before predicting.
                           # IMPORTANT: set this to match what your Flask app.py
                           # does. The model has no Rescaling layer inside.
LOW_CONFIDENCE = 0.60      # below this, show a "retake photo" warning

st.set_page_config(page_title="PlantCare AI - Disease Detection",
                   page_icon="🌱", layout="wide")

# --------------------------------------------------------------------------
# Styling (matches the Flask page)
# --------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, header[data-testid="stHeader"], footer {visibility: hidden; height: 0;}
.stApp {background: #ffffff;}
.block-container {padding: 0 !important; max-width: 100% !important;}

.pc-header {
    background: #1b6b3a; color: #fff; text-align: center;
    padding: 3.2rem 1rem 2.4rem;
}
.pc-header .pc-title {font-size: 3.2rem; font-weight: 800; line-height: 1.1;}
.pc-header .pc-sub {font-size: 1.25rem; margin-top: .8rem; opacity: .95;}

.st-key-card {
    max-width: 900px; margin: 3rem auto 2rem; background: #fff;
    border-radius: 26px; padding: 3rem 3.5rem 3.2rem;
    box-shadow: 0 8px 40px rgba(20, 60, 35, .12);
}
.pc-h2 {color: #1b6b3a; font-size: 2.3rem; font-weight: 800;
        text-align: center; margin: 0 0 .8rem;}
.pc-p  {color: #666; font-size: 1.2rem; text-align: center; margin-bottom: 2rem;}

.st-key-uploader {
    border: 2px dashed #2e8b57; border-radius: 20px;
    padding: 2rem 2rem 1.2rem; background: #f9fdfa;
}
.st-key-uploader [data-testid="stFileUploaderDropzone"] {
    background: #fff; border: 1px solid #dfe5e0; border-radius: 10px;
}

.st-key-detect {display: flex; justify-content: center; margin-top: .5rem;}
.st-key-detect button {
    background: #1b6b3a; border: none; border-radius: 12px;
    padding: .9rem 2.8rem; min-height: 3.4rem;
    box-shadow: 0 4px 12px rgba(27, 107, 58, .3);
}
.st-key-detect button p {color: #fff !important; font-size: 1.2rem; font-weight: 700;}
.st-key-detect button:hover {background: #145a30; border: none;}

.pc-result {border-radius: 16px; padding: 1.4rem 1.6rem; margin-top: 1.5rem;
            border: 1px solid #cfe8d7; background: #f1faf4;}
.pc-result.sick {background: #fff6f0; border-color: #f3d3bd;}
.pc-result .plant {color: #666; font-size: 1rem; text-transform: uppercase; letter-spacing: .06em;}
.pc-result .cond  {color: #1b6b3a; font-size: 1.8rem; font-weight: 800; margin: .2rem 0;}
.pc-result.sick .cond {color: #b4531a;}
.pc-result .conf  {color: #444; font-size: 1.05rem;}

.pc-bar-row {margin: .55rem 0;}
.pc-bar-label {display: flex; justify-content: space-between; font-size: .95rem; color: #444;}
.pc-bar {height: 8px; background: #e5ece7; border-radius: 6px; overflow: hidden; margin-top: 3px;}
.pc-bar > div {height: 100%; background: #2e8b57; border-radius: 6px;}

.pc-footer {text-align: center; color: #777; padding: 1.5rem 0 2.5rem; font-size: 1.05rem;}

@media (max-width: 700px) {
    .st-key-card {padding: 1.8rem 1.2rem; margin: 1.5rem .8rem;}
    .pc-header .pc-title {font-size: 2.2rem;}
    .pc-h2 {font-size: 1.7rem;}
}
</style>
"""

# --------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import keras  # imported lazily so the page shell renders fast
    return keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_class_names():
    with open(CLASSES_PATH, encoding="utf-8") as f:
        return json.load(f)


def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")
    if SCALE_TO_01:
        arr = arr / 255.0
    return np.expand_dims(arr, 0)


def predict(img: Image.Image, top_k: int = 3):
    model = load_model()
    names = load_class_names()
    probs = model.predict(preprocess(img), verbose=0)[0]
    idx = np.argsort(probs)[::-1][:top_k]
    return [(names[i], float(probs[i])) for i in idx]


def pretty(label: str):
    plant, _, cond = label.partition("___")
    return plant.replace("_", " "), cond.replace("_", " ")


def render_result(results):
    plant, cond = pretty(results[0][0])
    conf = results[0][1]
    healthy = cond.lower() == "healthy"
    st.markdown(
        f"""<div class="pc-result {'healthy' if healthy else 'sick'}">
              <div class="plant">{plant}</div>
              <div class="cond">{'✅ Healthy' if healthy else '⚠️ ' + cond}</div>
              <div class="conf">Confidence: <b>{conf*100:.1f}%</b></div>
            </div>""",
        unsafe_allow_html=True,
    )
    if conf < LOW_CONFIDENCE:
        st.warning("The model is not very sure about this one. Try a clearer, "
                   "well-lit photo of a single leaf against a plain background.")

    rows = ""
    for label, p in results:
        pl, cd = pretty(label)
        rows += (f'<div class="pc-bar-row"><div class="pc-bar-label">'
                 f'<span>{pl} - {cd}</span><span>{p*100:.1f}%</span></div>'
                 f'<div class="pc-bar"><div style="width:{p*100:.1f}%"></div></div></div>')
    st.markdown("<div style='margin-top:1.2rem;font-weight:700;color:#1b6b3a'>"
                "Top predictions</div>" + rows, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """<div class="pc-header">
             <div class="pc-title">🌱 PlantCare AI</div>
             <div class="pc-sub">Smart Plant Disease Detection</div>
           </div>""",
        unsafe_allow_html=True,
    )

    with st.container(key="card"):
        st.markdown('<div class="pc-h2">Detect Plant Diseases</div>'
                    '<div class="pc-p">Upload a clear picture of a plant leaf to '
                    'identify possible diseases.</div>', unsafe_allow_html=True)

        with st.container(key="uploader"):
            file = st.file_uploader("Choose a leaf image",
                                    type=["jpg", "jpeg", "png", "webp"],
                                    label_visibility="collapsed")
            if file is not None:
                img = Image.open(file)
                _, mid, _ = st.columns([1, 2, 1])
                mid.image(img, caption=file.name, use_container_width=True)

        clicked = st.button("Detect Disease", key="detect")

        if clicked:
            if file is None:
                st.warning("Please choose a leaf image first.")
            else:
                with st.spinner("Analysing leaf..."):
                    results = predict(img)
                render_result(results)

    st.markdown('<div class="pc-footer">PlantCare AI | AI-powered Plant Disease '
                'Detection</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()