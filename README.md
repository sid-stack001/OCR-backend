# OCR Microservice for Medical Reports

A containerized FastAPI backend designed to extract text from noisy, low-quality medical document scans. It uses a multi-stage OpenCV + OCR pipeline to handle shadows, table grids, and mixed font weights.

## Features

- OCR endpoint (`/ocr`) that returns cleaned text.
- Debug endpoint (`/debug-view`) that returns preprocessed image output.
- Otsu-based binarization to preserve bold text.
- Input validation for file type, empty files, and max file size.
- Configurable preprocessing parameters via environment variables.

## Requirements

- Python 3.10+
- Tesseract OCR installed and available on PATH
- Dependencies from `requirements.txt`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

## Environment variables

- `MAX_FILE_SIZE_BYTES` (default: `10485760`)
- `AI_MAX_WIDTH` (default: `1024`)
- `GAMMA` (default: `0.5`)
- `TARGET_WIDTH` (default: `2000`)
- `BLUR_KERNEL_SIZE` (default: `5`)
- `VERTICAL_KERNEL_WIDTH` (default: `1`)
- `VERTICAL_KERNEL_HEIGHT` (default: `40`)
- `LOG_LEVEL` (default: `INFO`)

## API

### `POST /ocr`

Multipart form with `file` image.

Success response:

```json
{
  "status": "success",
  "text": "..."
}
```

Validation errors return `400`; processing errors return `500`.

### `POST /debug-view`

Multipart form with `file` image. Returns `image/jpeg` of the processed document.

## Testing

```bash
pytest -q
```
