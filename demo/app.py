"""Streamlit demo for the original 40-attribute CelebA CNN.

Run from the repository root:
    streamlit run demo/app.py

This app predicts *facial attributes*, not a person's identity.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Make root-level predict.py importable from demo/ on Windows and Unix.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict import CELEBA_ATTRIBUTES, DEFAULT_MODEL, preprocess_image_bytes  # noqa: E402

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000

# Disjoint groups, preserving the names used in the original CelebA dataset.
GROUPS = {
    "Expression & appearance": [
        "Smiling", "Mouth_Slightly_Open", "High_Cheekbones", "Rosy_Cheeks",
        "Heavy_Makeup", "Wearing_Lipstick", "Attractive", "Blurry", "Young",
    ],
    "Hair & facial hair": [
        "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
        "Gray_Hair", "Receding_Hairline", "Straight_Hair", "Wavy_Hair",
        "5_o_Clock_Shadow", "Goatee", "Mustache", "No_Beard", "Sideburns",
    ],
    "Facial features": [
        "Arched_Eyebrows", "Bags_Under_Eyes", "Big_Lips", "Big_Nose",
        "Bushy_Eyebrows", "Chubby", "Double_Chin", "Male", "Narrow_Eyes",
        "Oval_Face", "Pale_Skin", "Pointy_Nose",
    ],
    "Accessories": [
        "Eyeglasses", "Wearing_Earrings", "Wearing_Hat", "Wearing_Necklace",
        "Wearing_Necktie",
    ],
}
assert set(attr for attrs in GROUPS.values() for attr in attrs) == set(CELEBA_ATTRIBUTES)
assert sum(map(len, GROUPS.values())) == 40

st.set_page_config(
    page_title="Facial Attribute Analyzer | CNN CelebA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Manrope:wght@400;500;600&display=swap');
    :root { --ink:#151581; --lavender:#7777a5; --porcelain:#f6f6fa; --paper:#ffffff;
            --fog:#d6d6d6; --success:#00bb76; --shadow:0 8px 34px rgba(36,36,41,.07),0 1px 2px rgba(37,36,41,.09); }
    html,body,[class*="css"] {font-family:'Inter',sans-serif;}
    .stApp {background:var(--porcelain); color:#292824;}
    /* Streamlit toolbar overlays the document flow. Reserve space on ALL breakpoints. */
    div.block-container {max-width:1200px; padding-top:6.5rem !important; padding-bottom:5rem;}
    header[data-testid="stHeader"] {background:rgba(246,246,250,.96); height:3.75rem;}
    [data-testid="stToolbar"] {right:1rem;}
    h1,h2,h3 {font-family:'Manrope','Inter',sans-serif !important;color:var(--ink) !important;
      font-weight:400 !important;letter-spacing:-.035em !important;line-height:1.15 !important;}
    h2 {font-size:1.65rem !important;}
    h3 {font-size:1.25rem !important;}
    p,span,label {line-height:1.4;}
    .hero {text-align:center;margin:0 auto 3rem; max-width:880px;}
    .hero-kicker {display:inline-block;border:1px solid #e6e5ee;background:white;border-radius:999px;
      padding:8px 18px;font-size:.72rem;letter-spacing:.11em;font-weight:600;color:var(--ink);margin-bottom:1.5rem;}
    .hero-title {font-family:'Manrope','Inter',sans-serif;font-weight:400;color:var(--ink);
      font-size:clamp(2.25rem,5vw,3.6rem);letter-spacing:-.05em;line-height:1.12;margin:0 0 1rem;}
    .hero-subtitle {max-width:730px;margin:auto;color:#7777a5;line-height:1.6;font-size:1.05rem;}
    div[data-testid="stSidebar"] {background:#fff; border-right:1px solid #ededf2;}
    div[data-testid="stSidebar"] h3 {color:var(--ink);}
    div[data-testid="stFileUploader"] {background:#fff;border-radius:24px;padding:15px 22px;
       box-shadow:var(--shadow);}
    [data-testid="stFileUploaderDropzone"] {background:#f9f9fc;border:1px dashed #c9c7df;
       border-radius:16px;}
    [data-testid="stMetric"] {background:#fff; border:none !important; border-radius:20px;
       padding:18px !important; box-shadow:var(--shadow);min-height:100px;}
    [data-testid="stMetricValue"] {color:var(--ink);font-family:'Manrope','Inter',sans-serif;}
    [data-testid="stImage"] img {border-radius:16px;}
    [data-testid="stProgressBar"] > div {background:#ecebf5;border-radius:12px;}
    [data-testid="stProgressBar"] > div > div {background:#151581;border-radius:12px;}
    .stTabs [data-baseweb="tab-list"] {gap:12px;}
    .stTabs [data-baseweb="tab"] {border-radius:40px;padding:8px 18px;color:var(--ink);}
    .stTabs [aria-selected="true"] {background:#fff;}
    div[data-testid="stExpander"] {background:white;border:none;border-radius:24px;
      box-shadow:0 8px 34px rgba(36,36,41,.035);overflow:hidden;}
    .stButton button,.stDownloadButton button {border-radius:48px !important;background:#fff !important;
      color:var(--ink) !important;border:1px solid #dedde9 !important;}
    .stButton button:hover,.stDownloadButton button:hover {border-color:var(--ink) !important;}
    a {color:var(--ink);}
    @media(max-width:800px){
      div.block-container {padding:5.8rem 1rem 3rem !important;}
      .hero {margin-bottom:2rem;}
      .hero-kicker {font-size:.62rem;letter-spacing:.05em;padding:8px 10px;}
      .hero-subtitle {font-size:.92rem;}
    }
    </style>""",
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading the original CelebA CNN…")
def load_cnn(model_file: str):
    # Import here so Streamlit can render an informative missing-dependency message.
    import tensorflow as tf

    cnn = tf.keras.models.load_model(model_file, compile=False)
    if tuple(cnn.input_shape[1:]) != (128, 128, 3):
        raise ValueError(f"Unexpected input shape: {cnn.input_shape}")
    if cnn.output_shape[-1] != len(CELEBA_ATTRIBUTES):
        raise ValueError(
            "This checkpoint is not the 40-attribute CelebA CNN "
            f"(model outputs {cnn.output_shape[-1]} values)."
        )
    return cnn


@st.cache_data(show_spinner="Analyzing image…", max_entries=12)
def infer(image_bytes: bytes, model_file: str, model_mtime: float):
    # Cache key contains bytes and model modification time; threshold doesn't rerun CNN.
    del model_mtime
    model = load_cnn(model_file)
    batch = preprocess_image_bytes(image_bytes)
    raw = np.asarray(model(batch, training=False))[0]
    if raw.shape != (40,) or not np.all(np.isfinite(raw)):
        raise ValueError("The model returned unexpected probabilities.")
    if np.any((raw < 0) | (raw > 1)):
        raise ValueError("The model output is outside the [0, 1] range.")
    return [float(v) for v in raw]


def readable(label: str) -> str:
    return label.replace("_", " ")


def show_scores(items: list[tuple[str, float]], threshold: float) -> None:
    for name, prob in items:
        left, right = st.columns([4, 1])
        with left:
            mark = "✓" if prob >= threshold else "·"
            st.markdown(f"**{mark} {readable(name)}**")
        with right:
            st.markdown(f"**{prob:.1%}**")
        st.progress(prob)


st.markdown(
    """<section class="hero">
      <div class="hero-kicker">COMPUTER VISION / DEEP LEARNING / CELEBA</div>
      <h1 class="hero-title">Facial Attribute Analyzer</h1>
      <p class="hero-subtitle">Explore 40 facial attributes with a custom convolutional neural
      network trained on CelebA. A research demonstration of attribute prediction,
      <strong>not identity recognition.</strong></p>
    </section>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Analysis settings")
    threshold = st.slider(
        "Classification threshold", min_value=0.0, max_value=1.0,
        value=0.50, step=0.05,
        help="A score at or above the threshold is marked as detected. "
             "Changing it does not modify model probabilities.",
    )
    top_k = st.slider("Top attributes", 5, 40, 10, 5)
    show_all = st.toggle("Show all 40 attributes", value=False)
    st.divider()
    st.caption("MODEL")
    st.code("celebA_grande_mejor.h5", language=None)
    st.caption("INPUT  ·  128 × 128 RGB")
    st.caption("OUTPUT · 40 sigmoid scores")
    st.caption("TensorFlow / Keras")

model_path = DEFAULT_MODEL
if not model_path.is_file():
    st.error(
        "Model checkpoint not found. Expected `Modelo CNN celebA/celebA_grande_mejor.h5` "
        "relative to the repository root."
    )
    st.stop()

uploaded = st.file_uploader(
    "Upload a portrait (JPEG or PNG, up to 10 MB)", type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Use an image you own or have permission to analyze. "
         "Images stay in the current Streamlit session; no image is saved by this app.",
)
if uploaded is None:
    st.info("Upload a portrait to see the 40 CNN predictions.")
    st.caption("Best results are expected for a single, frontal, aligned face similar to CelebA.")
    st.stop()

image_bytes = uploaded.getvalue()
if len(image_bytes) > MAX_FILE_BYTES:
    st.error("Image is too large. Please use a file smaller than 10 MB.")
    st.stop()

try:
    with Image.open(io.BytesIO(image_bytes)) as check:
        if check.format not in {"JPEG", "PNG"}:
            raise ValueError("Only JPEG or PNG images are accepted.")
        if check.width * check.height > MAX_PIXELS:
            raise ValueError("Image dimensions are too large (maximum: 20 megapixels).")
        preview = check.convert("RGB")
except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
    st.error(f"Cannot read image: {exc}")
    st.stop()

try:
    with st.spinner("Running the CNN on your image…"):
        scores = infer(image_bytes, str(model_path), model_path.stat().st_mtime)
except (ImportError, OSError, ValueError, RuntimeError, TypeError) as exc:
    st.error(f"Could not run inference: {exc}")
    st.stop()

results = list(zip(CELEBA_ATTRIBUTES, scores))
ordered = sorted(results, key=lambda row: row[1], reverse=True)
detected = sum(p >= threshold for _, p in results)

left, right = st.columns([0.9, 1.35], gap="large")
with left:
    st.subheader("Input image")
    st.image(preview, caption="Uploaded portrait", use_container_width=True)
    st.caption(
        "The model resizes input to 128 × 128 and scales RGB pixel values to [0, 1]. "
        "No automatic face alignment or detection is performed."
    )
with right:
    st.subheader("Model output")
    m1, m2, m3 = st.columns(3)
    m1.metric("Attributes", "40")
    m2.metric("Above threshold", str(detected))
    m3.metric("Threshold", f"{threshold:.0%}")
    st.caption("Scores are uncalibrated model estimates, not verified facts about a person.")
    st.markdown(f"#### Top {top_k} predicted attributes")
    show_scores(ordered[:top_k], threshold)

st.divider()
tab_group, tab_all, tab_about = st.tabs(["By category", "All attributes", "About the model"])

with tab_group:
    for title, names in GROUPS.items():
        subset = sorted(((n, dict(results)[n]) for n in names), key=lambda x: x[1], reverse=True)
        with st.expander(f"{title}  ·  {sum(p >= threshold for _, p in subset)}/{len(subset)} detected", expanded=True):
            show_scores(subset if show_all else [r for r in subset if r[1] >= threshold], threshold)
            if not show_all and not any(p >= threshold for _, p in subset):
                st.caption("No attributes in this category reach the selected threshold.")

with tab_all:
    st.dataframe(
        [
            {"Attribute": readable(name), "Probability": round(prob, 5), "Detected": prob >= threshold}
            for name, prob in ordered
        ],
        use_container_width=True,
        hide_index=True,
        column_config={"Probability": st.column_config.ProgressColumn(
            "Probability", min_value=0.0, max_value=1.0, format="%.3f"
        )},
    )

with tab_about:
    st.markdown(
        """The original model was trained from scratch to predict **40 CelebA attributes**.
        It uses a convolutional backbone and a sigmoid output per attribute; the
        **transfer-learning identity classifiers are not used in this demo**.

        **Limitations:** CelebA reflects its own annotations and biases. Labels such as
        *Attractive*, *Male*, and *Young* are dataset categories, not objective
        statements about a person. High probabilities do not imply calibrated
        confidence. Performance can deteriorate on uncropped, off-angle, or
        out-of-distribution photos.

        **Privacy:** This program does not persist uploads to disk or send them to
        an external model service. Running it on a public Streamlit host would
        still transfer uploaded files to that host; obtain consent before use.

        **Label order:** The 40 output labels use the standard CelebA order. They
        should be checked against the original training CSV if available.
        """
    )

payload = {
    "model": model_path.name,
    "threshold": threshold,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "note": "CelebA model predictions; not verified facts or identity recognition.",
    "attributes": [
        {"attribute": name, "probability": prob, "predicted_present": prob >= threshold}
        for name, prob in results
    ],
}
st.download_button(
    "Download results (JSON)",
    data=json.dumps(payload, indent=2, ensure_ascii=False),
    file_name="facial_attributes.json",
    mime="application/json",
    use_container_width=False,
)
