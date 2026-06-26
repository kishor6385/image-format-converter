"""
Image Read/Write - OpenCV fundamentals with a polished developer experience.

Load an image, preview it with metadata, and persist it in a new format.
Run without arguments to use a built-in sample image - zero setup required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_READ = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SUPPORTED_WRITE = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def create_sample_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a vibrant gradient sample when no input file is provided."""
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)

    blue = xx
    green = yy
    red = ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)

    image = np.dstack([blue, green, red])

    cv2.putText(
        image,
        "OpenCV Sample",
        (width // 2 - 140, height // 2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Press any key to continue",
        (width // 2 - 180, height // 2 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return image


def load_image(path: Path | None) -> tuple[np.ndarray, str]:
    """Load from disk or fall back to an in-memory sample."""
    if path is None:
        print("No input provided - generating a sample image.")
        return create_sample_image(), "in-memory sample"

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_READ:
        print(f"Warning: '{suffix}' may not be fully supported by OpenCV.")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode: {path}")

    return image, str(path)


def get_image_info(image: np.ndarray, source: str) -> dict[str, str | int]:
    """Return image metadata as a dictionary."""
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    return {
        "source": source,
        "width": w,
        "height": h,
        "channels": channels,
        "depth": str(image.dtype),
        "dimensions": f"{w} x {h} px",
    }


def describe_image(image: np.ndarray, source: str) -> None:
    """Print a concise, human-friendly summary."""
    info = get_image_info(image, source)
    print()
    print("  Image loaded successfully")
    print(f"  Source     : {info['source']}")
    print(f"  Dimensions : {info['dimensions']}")
    print(f"  Channels   : {info['channels']}")
    print(f"  Depth      : {info['depth']}")
    print()


def decode_image_bytes(data: bytes, source: str = "upload") -> np.ndarray:
    """Decode raw image bytes with OpenCV."""
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode: {source}")
    return image


def encode_image(image: np.ndarray, target_format: str, jpeg_quality: int = 95) -> tuple[bytes, str, str]:
    """Encode an image to bytes. Returns (data, mime_type, extension)."""
    ext = target_format if target_format.startswith(".") else f".{target_format}"
    ext = ext.lower()

    if ext not in SUPPORTED_WRITE:
        raise ValueError(f"Unsupported output format: {ext}")

    params: list[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        mime = "image/jpeg"
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        mime = "image/png"
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, jpeg_quality]
        mime = "image/webp"
    elif ext in {".tif", ".tiff"}:
        mime = "image/tiff"
    else:
        mime = "image/bmp"

    success, buffer = cv2.imencode(ext, image, params)
    if not success:
        raise IOError(f"Failed to encode image as {ext}")

    return buffer.tobytes(), mime, ext


def display_image(image: np.ndarray, window_title: str = "OpenCV - Preview") -> None:
    """Show the image and wait for a keypress."""
    cv2.imshow(window_title, image)
    print("Preview open - press any key in the image window to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resolve_output_path(
    input_path: Path | None,
    output: Path | None,
    target_format: str,
) -> Path:
    """Derive an output path with a different extension when not explicitly set."""
    ext = target_format if target_format.startswith(".") else f".{target_format}"
    ext = ext.lower()

    if ext not in SUPPORTED_WRITE:
        raise ValueError(
            f"Unsupported output format '{ext}'. "
            f"Choose from: {', '.join(sorted(SUPPORTED_WRITE))}"
        )

    if output is not None:
        out = output
        if out.suffix == "":
            out = out.with_suffix(ext)
        return out

    if input_path is not None:
        stem = input_path.stem
        parent = input_path.parent
    else:
        stem = "sample_output"
        parent = Path("output")

    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{stem}_converted{ext}"


def save_image(image: np.ndarray, output_path: Path, jpeg_quality: int = 95) -> None:
    """Write the image, applying format-specific encoder options."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    params: list[int] = []

    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, jpeg_quality]

    success = cv2.imwrite(str(output_path), image, params)
    if not success:
        raise IOError(f"Failed to write image to: {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load, preview, and save images with OpenCV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python image_read_write.py\n"
            "  python image_read_write.py photo.png --format jpg\n"
            "  python image_read_write.py input.bmp -o results/photo.webp\n"
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to the source image (optional - a sample is generated if omitted)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Destination file path (extension inferred from --format if missing)",
    )
    parser.add_argument(
        "-f", "--format",
        default="png",
        help="Target format when converting (default: png)",
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=95,
        help="JPEG/WebP quality 0–100 (default: 95)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the preview window",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        image, source = load_image(args.input)
        describe_image(image, source)

        if not args.no_display:
            display_image(image)

        output_path = resolve_output_path(args.input, args.output, args.format)

        input_ext = args.input.suffix.lower() if args.input else ".generated"
        output_ext = output_path.suffix.lower()

        save_image(image, output_path, jpeg_quality=args.quality)

        print(f"Saved as {output_ext} -> {output_path.resolve()}")
        if input_ext != output_ext:
            print(f"Format conversion: {input_ext or '(sample)'} -> {output_ext}")
        else:
            print("Note: input and output share the same extension.")

        print("Done.")
        return 0

    except (FileNotFoundError, ValueError, IOError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
