# Facial Attribute Analyzer — Streamlit demo

Interactive demo of the **original custom CelebA CNN** (40 attribute outputs). No identity-recognition checkpoints are required.

## Design

The interface uses the supplied [`DESIGN.md`](../DESIGN.md) reference: porcelain canvas, indigo ink, restrained white cards, rounded shapes, and Manrope/Inter typography. Manrope is a free substitute for the reference's proprietary display font. The hero reserves sufficient space below the Streamlit toolbar on desktop and mobile, keeping the `COMPUTER VISION / DEEP LEARNING / CELEBA` label visible.

## Structure

```text
Reconocimiento-Facial/
├── .streamlit/config.toml
├── demo/app.py
├── demo/README-demo.md
├── predict.py
├── requirements-demo.txt
├── DESIGN.md
└── Modelo CNN celebA/celebA_grande_mejor.h5  # existing checkpoint
```

## Run locally (Windows, Python 3.13)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-demo.txt
python -m streamlit run demo/app.py
```

To open the demo, visit `http://localhost:8501`.

## Features

- Upload a JPEG or PNG portrait (10 MB and 20 MP limits).
- Browse top scores and 40 predictions, grouped by category.
- Change the classification threshold without rerunning the CNN.
- Export all predicted scores as JSON.
- Preview works without saving images to disk.

## Scientific and privacy notes

This model predicts **dataset attributes**, not identity or factual characteristics. CelebA labels may contain subjective definitions and biases. It expects a single aligned face, RGB 128 × 128 and `[0, 1]` scaling; no detection/alignment is included. Prediction-to-label mapping assumes the standard CelebA CSV column order and should be verified against the original CSV if it is recovered. On remote hosting, user uploads are transmitted to the server, so obtain consent before using any person's photos. Do not commit personal images or identity classifier checkpoints.
