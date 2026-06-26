"""Web UI for Image Read/Write with OpenCV."""

from __future__ import annotations

import base64
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from io import BytesIO

from image_read_write import (
    SUPPORTED_WRITE,
    create_sample_image,
    decode_image_bytes,
    encode_image,
    get_image_info,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


def image_to_data_url(image_bytes: bytes, mime: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@app.route("/")
def index():
    formats = sorted(ext.lstrip(".") for ext in SUPPORTED_WRITE)
    return render_template("index.html", formats=formats)


@app.route("/api/sample", methods=["POST"])
def load_sample():
    image = create_sample_image()
    data, mime, ext = encode_image(image, "png")
    info = get_image_info(image, "Built-in sample")
    return jsonify({
        "preview": image_to_data_url(data, mime),
        "info": info,
        "input_format": "sample",
        "image_b64": base64.b64encode(data).decode("ascii"),
        "image_mime": mime,
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    raw = file.read()
    if not raw:
        return jsonify({"error": "File is empty."}), 400

    try:
        image = decode_image_bytes(raw, file.filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    suffix = Path(file.filename).suffix.lower() or ".unknown"
    preview_bytes, mime, _ = encode_image(image, "png")
    info = get_image_info(image, file.filename)

    return jsonify({
        "preview": image_to_data_url(preview_bytes, mime),
        "info": info,
        "input_format": suffix.lstrip("."),
        "image_b64": base64.b64encode(raw).decode("ascii"),
        "image_mime": mime,
    })


@app.route("/api/convert", methods=["POST"])
def convert():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image_b64")
    target_format = payload.get("format", "png")
    quality = int(payload.get("quality", 95))
    filename = payload.get("filename", "converted")

    if not image_b64:
        return jsonify({"error": "No image data to convert."}), 400

    try:
        raw = base64.b64decode(image_b64)
        image = decode_image_bytes(raw, filename)
        converted, mime, ext = encode_image(image, target_format, jpeg_quality=quality)
    except (ValueError, IOError) as exc:
        return jsonify({"error": str(exc)}), 400

    stem = Path(filename).stem or "image"
    download_name = f"{stem}_converted{ext}"

    return jsonify({
        "preview": image_to_data_url(converted, mime),
        "download_name": download_name,
        "output_format": ext.lstrip("."),
        "size_kb": round(len(converted) / 1024, 1),
        "file_b64": base64.b64encode(converted).decode("ascii"),
        "mime": mime,
    })


@app.route("/api/download", methods=["POST"])
def download():
    payload = request.get_json(silent=True) or {}
    file_b64 = payload.get("file_b64")
    download_name = payload.get("download_name", "converted.png")
    mime = payload.get("mime", "application/octet-stream")

    if not file_b64:
        return jsonify({"error": "Nothing to download."}), 400

    data = base64.b64decode(file_b64)
    return send_file(
        BytesIO(data),
        mimetype=mime,
        as_attachment=True,
        download_name=download_name,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
