#!/usr/bin/env python3
"""Predict 40 CelebA facial attributes using the original trained CNN.

Example:
    python predict.py --image example.jpg --top 10

The ordering below follows the standard CelebA attribute CSV. The training script
used ``df.columns[1:]``, so verify the original CSV columns if available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# IMPORTANT: output index -> CSV column mapping must match training exactly.
CELEBA_ATTRIBUTES = (
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
)

DEFAULT_MODEL = Path(__file__).resolve().parent / "Modelo CNN celebA" / "celebA_grande_mejor.h5"


def verify_csv_order(csv_path: Path) -> None:
    """Require original attribute CSV to match the declared model output order."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    actual = tuple(header[1:]) if header and header[0].strip() == "image_id" else tuple(header)
    if actual != CELEBA_ATTRIBUTES:
        raise ValueError(
            "Attribute order in CSV does not match predict.py. "
            "Do not use the current labels until the original column order "
            "has been checked."
        )


def preprocess_image_bytes(image_bytes: bytes):
    """Use the same RGB decode, bilinear resize and scaling as training."""
    import tensorflow as tf

    rgb = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    rgb.set_shape([None, None, 3])
    resized = tf.image.resize(rgb, [128, 128])
    return tf.expand_dims(tf.cast(resized, tf.float32) / 255.0, axis=0)


def predict(image_path: Path, model_path: Path) -> list[dict[str, object]]:
    # Imported lazily so --help works without TensorFlow installed.
    import numpy as np
    import tensorflow as tf

    # compile=False avoids needing the original optimizer state for inference.
    model = tf.keras.models.load_model(str(model_path), compile=False)
    if model.output_shape[-1] != len(CELEBA_ATTRIBUTES):
        raise ValueError(
            f"This model produces {model.output_shape[-1]} outputs; expected 40. "
            "Use the CelebA attribute model, not a binary transfer-learning model."
        )
    if tuple(model.input_shape[1:]) != (128, 128, 3):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")

    batch = preprocess_image_bytes(image_path.read_bytes())
    probabilities = np.asarray(model(batch, training=False))[0]
    if len(probabilities) != 40 or not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned invalid attribute probabilities.")
    return [
        {"attribute": label, "probability": float(prob)}
        for label, prob in zip(CELEBA_ATTRIBUTES, probabilities)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict 40 CelebA facial attributes from a photograph."
    )
    parser.add_argument("--image", required=True, type=Path, help="JPEG/PNG image path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="CelebA .h5 checkpoint path")
    parser.add_argument("--top", type=int, default=10, help="Number of highest scoring attributes (0 = all)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default 0.5)")
    parser.add_argument("--json", dest="json_path", type=Path, help="Write all 40 scores to a JSON file")
    parser.add_argument("--attributes-csv", type=Path, help="Validate original CSV attribute column order")
    args = parser.parse_args()

    if not (0 <= args.threshold <= 1):
        parser.error("--threshold must be between 0 and 1")
    if args.top < 0:
        parser.error("--top must be 0 or greater")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")

    try:
        if args.attributes_csv:
            verify_csv_order(args.attributes_csv)
        rows = predict(args.image, args.model)
        ordered = sorted(rows, key=lambda item: item["probability"], reverse=True)
        shown = ordered if args.top == 0 else ordered[: args.top]
        print(f"\nImage: {args.image}\nModel: {args.model}")
        print(f"Threshold: {args.threshold:.2f}\n")
        for item in shown:
            p = item["probability"]
            status = "YES" if p >= args.threshold else "no"
            print(f"{item['attribute']:<25} {p:>7.2%}  {status}")
        print("\nScores are model estimates, not verified facts about a person.")
        if args.json_path:
            output = {
                "image": str(args.image),
                "model": str(args.model),
                "threshold": args.threshold,
                "attributes": [
                    {**item, "predicted_present": item["probability"] >= args.threshold}
                    for item in rows
                ],
            }
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"Saved JSON: {args.json_path}")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
