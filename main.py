"""
OCR Backend Service - FastAPI Application.

A containerized backend that preprocesses document images and extracts text with
Tesseract OCR.
"""

import gc
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from rembg import new_session, remove

logger = logging.getLogger("ocr_backend")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


@dataclass(frozen=True)
class Settings:
    max_file_size_bytes: int = int(os.getenv("MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024))
    ai_max_width: int = int(os.getenv("AI_MAX_WIDTH", 1024))
    gamma: float = float(os.getenv("GAMMA", 0.5))
    target_width: int = int(os.getenv("TARGET_WIDTH", 2000))
    blur_kernel_size: int = int(os.getenv("BLUR_KERNEL_SIZE", 5))
    vertical_kernel_width: int = int(os.getenv("VERTICAL_KERNEL_WIDTH", 1))
    vertical_kernel_height: int = int(os.getenv("VERTICAL_KERNEL_HEIGHT", 40))


SETTINGS = Settings()
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/tiff",
    "image/bmp",
}


if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Initializing rembg session")
    app.state.rembg_session = new_session("u2netp")
    logger.info("rembg session initialized")
    yield
    logger.info("Shutting down OCR backend")


app = FastAPI(lifespan=lifespan)


class ValidationError(Exception):
    """Raised for client-side validation issues."""


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def remove_vertical_lines(image: np.ndarray) -> np.ndarray:
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (SETTINGS.vertical_kernel_width, SETTINGS.vertical_kernel_height),
    )
    detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
    return cv2.add(image, detected_lines)


def clean_text_for_llm(raw_text: str) -> str:
    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        clean = line.replace("|", " ").replace("_", " ").replace("—", " ")
        clean = re.sub(r"\[\s*\]", "", clean)
        clean = re.sub(r"\(\s*\)", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) > 2:
            cleaned_lines.append(clean)
    return "\n".join(cleaned_lines)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    perspective_matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, perspective_matrix, (max_width, max_height))


def advanced_preprocess(img_array: np.ndarray) -> np.ndarray:
    original = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if original is None:
        raise ValidationError("Image decoding failed")

    scale_factor = 1.0
    h, w = original.shape[:2]
    if w > SETTINGS.ai_max_width:
        scale_factor = SETTINGS.ai_max_width / w
        input_for_ai = cv2.resize(original, (SETTINGS.ai_max_width, int(h * scale_factor)))
    else:
        input_for_ai = original

    warped = original

    try:
        pil_img = Image.fromarray(cv2.cvtColor(input_for_ai, cv2.COLOR_BGR2RGB))
        no_bg = remove(pil_img, session=app.state.rembg_session)

        del pil_img
        gc.collect()

        alpha = np.array(no_bg)[:, :, 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            min_area = 0.10 * (input_for_ai.shape[0] * input_for_ai.shape[1])
            if cv2.contourArea(contour) > min_area:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                if scale_factor != 1.0:
                    approx = (approx / scale_factor).astype(np.float32)

                if len(approx) == 4:
                    warped = four_point_transform(original, approx.reshape(4, 2))
                else:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    if scale_factor != 1.0:
                        box = (box / scale_factor).astype(np.float32)
                    warped = four_point_transform(original, np.int32(box))
    except (cv2.error, ValueError, IndexError) as exc:
        logger.warning("Document alignment fallback activated: %s", exc)
        warped = original

    gc.collect()

    warped = adjust_gamma(warped, gamma=SETTINGS.gamma)

    scale = SETTINGS.target_width / warped.shape[1]
    if scale > 1:
        resized = (SETTINGS.target_width, int(warped.shape[0] * scale))
        warped = cv2.resize(warped, resized, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    kernel = SETTINGS.blur_kernel_size
    if kernel % 2 == 0:
        kernel += 1
    gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    clean = remove_vertical_lines(binary)
    clean = cv2.copyMakeBorder(clean, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return clean


async def read_and_validate_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise ValidationError(f"Unsupported content-type: {file.content_type}")

    content = await file.read()
    if not content:
        raise ValidationError("Uploaded file is empty")
    if len(content) > SETTINGS.max_file_size_bytes:
        raise ValidationError(
            f"File too large. Max size is {SETTINGS.max_file_size_bytes // (1024 * 1024)}MB"
        )
    return content


@app.post("/ocr")
async def get_ocr(file: UploadFile = File(...)):
    try:
        contents = await read_and_validate_upload(file)
        nparr = np.frombuffer(contents, np.uint8)
        processed_img = advanced_preprocess(nparr)

        custom_config = r"--oem 1 --psm 6"
        raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
        clean_text = clean_text_for_llm(raw_text)

        del processed_img
        del nparr
        gc.collect()

        return {"status": "success", "text": clean_text}
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled OCR error")
        raise HTTPException(status_code=500, detail="OCR processing failed") from exc


@app.post("/debug-view")
async def debug_view(file: UploadFile = File(...)):
    try:
        contents = await read_and_validate_upload(file)
        nparr = np.frombuffer(contents, np.uint8)
        processed_img = advanced_preprocess(nparr)
        ok, encoded_img = cv2.imencode(".jpg", processed_img)
        if not ok:
            raise RuntimeError("Failed to encode debug image")
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled debug-view error")
        raise HTTPException(status_code=500, detail="Debug processing failed") from exc


@app.get("/")
def home():
    return {"message": "✅ OCR Backend Service - v4.0 (Otsu Edition)"}
