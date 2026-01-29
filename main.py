"""
OCR Backend Service - FastAPI Application (Low Memory Edition)

This module implements an optimized OCR pipeline designed for cloud environments
with limited RAM (e.g., Railway Starter Tier). It features "Smart Downscaling" 
for AI operations and aggressive garbage collection to prevent OOM errors.

Author: Siddharth Verma
Version: 2.0 (Low-RAM Optimization)
"""

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from rembg import remove, new_session
from PIL import Image
import io
import os
import re
import gc  # Explicit garbage collection for memory management

app = FastAPI()

# Initialize background removal model (u2netp is the lightweight version)
session = new_session("u2netp")

# Configure Tesseract OCR path for Windows environments
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def adjust_gamma(image, gamma=1.0):
    """
    Apply gamma correction to adjust image brightness and contrast.
    
    Args:
        image (np.ndarray): Input image (grayscale or color)
        gamma (float): Gamma value. <1.0 darkens (bolds text), >1.0 lightens.
    
    Returns:
        np.ndarray: Gamma-corrected image
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def remove_vertical_lines(image):
    """
    Remove vertical table lines while preserving text integrity.
    
    Updated for v8.0: Kernel height reduced to 40px to match the lower 
    target resolution (2000px width).
    
    Args:
        image (np.ndarray): Binary image (text on white background)
    
    Returns:
        np.ndarray: Image with vertical lines removed
    """
    # Kernel: 1px wide, 40px tall (Detects vertical structures)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
    result = cv2.add(image, detected_lines)
    return result


def clean_text_for_llm(raw_text):
    """
    Clean OCR output text for downstream LLM processing.
    
    Removes visual noise artifacts (grid lines, empty brackets) that 
    confuse language models.
    
    Args:
        raw_text (str): Raw OCR output from Tesseract
    
    Returns:
        str: Cleaned text ready for LLM processing
    """
    lines = raw_text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Replace common table characters with space
        clean = line.replace('|', ' ').replace('_', ' ').replace('—', ' ')
        # Remove empty checkboxes/brackets
        clean = re.sub(r'\[\s*\]', '', clean)
        clean = re.sub(r'\(\s*\)', '', clean)
        # Collapse whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Filter out tiny noise lines (less than 3 chars)
        if len(clean) > 2:
            cleaned_lines.append(clean)
    return "\n".join(cleaned_lines)


def order_points(pts):
    """
    Order corner points: top-left, top-right, bottom-right, bottom-left.
    Essential for correct perspective transformation.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
    Apply perspective transformation to flatten the document.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


# ============================================================================
# IMAGE PREPROCESSING PIPELINE (v8.0 - Low Memory)
# ============================================================================

def advanced_preprocess(img_array):
    """
    Execute optimized image preprocessing pipeline for OCR.
    
    v8.0 Optimization Strategy:
    - Downscale input for AI detection (massive RAM savings)
    - Upscale output to 2000px (instead of 3000px)
    - Aggressive variable deletion and garbage collection
    
    Pipeline stages:
    1. Smart Downscaling & AI Crop
    2. Gamma Correction
    3. Moderate Upscaling (2000px)
    4. Bilateral Filtering
    5. Adaptive Thresholding
    6. Vertical Line Removal
    7. Padding
    
    Args:
        img_array (np.ndarray): Raw image bytes
    
    Returns:
        np.ndarray: Processed binary image
    """
    original = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if original is None: 
        raise ValueError("Image decoding failed")

    # --- STAGE 1: Memory-Optimized AI Detection ---
    # Problem: U-2-Net uses massive RAM on 4K images.
    # Solution: Resize input to 1024px width for detection only.
    scale_factor = 1.0
    h, w = original.shape[:2]
    
    if w > 1024:
        scale_factor = 1024 / w
        input_for_ai = cv2.resize(original, (1024, int(h * scale_factor)))
    else:
        input_for_ai = original

    warped = original
    
    try:
        # Run U-2-Net on the smaller image
        pil_img = Image.fromarray(cv2.cvtColor(input_for_ai, cv2.COLOR_BGR2RGB))
        no_bg = remove(pil_img, session=session)
        
        # Free memory immediately after AI step
        del pil_img
        gc.collect()

        alpha = np.array(no_bg)[:, :, 3]
        cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            # Check if contour is significant (>10% of image area)
            if cv2.contourArea(c) > (0.10 * (input_for_ai.shape[0]*input_for_ai.shape[1])):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                # Scale contours back up to original resolution
                if scale_factor != 1.0:
                    approx = (approx / scale_factor).astype(np.float32)

                if len(approx) == 4:
                    warped = four_point_transform(original, approx.reshape(4, 2))
                else:
                    # Fallback to bounding box if 4 corners not found
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    if scale_factor != 1.0:
                        box = (box / scale_factor).astype(np.float32)
                    box = np.int32(box)
                    warped = four_point_transform(original, box)
    except Exception as e:
        print(f"Warning: AI Crop failed ({e}), using original image.")
        warped = original

    # Explicit cleanup
    gc.collect()

    # --- STAGE 2: Gamma Correction ---
    # 0.5 Darkens mid-tones to make faint text (thin fonts) bold
    warped = adjust_gamma(warped, gamma=0.5)

    # --- STAGE 3: Moderate Upscaling ---
    # Reduced from 3000px -> 2000px to prevent OOM on Cloud Free Tier
    target_width = 2000
    scale = target_width / warped.shape[1]
    if scale > 1:
        dim = (target_width, int(warped.shape[0] * scale))
        warped = cv2.resize(warped, dim, interpolation=cv2.INTER_CUBIC)

    # --- STAGE 4: Denoising ---
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # --- STAGE 5: Adaptive Thresholding ---
    # Block Size adjusted to 51 (from 61) due to lower resolution (2000px)
    binary = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        51, 15 
    )

    # --- STAGE 6: Vertical Line Removal ---
    clean = remove_vertical_lines(binary)

    # --- STAGE 7: Padding ---
    clean = cv2.copyMakeBorder(clean, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return clean


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/ocr")
async def get_ocr(file: UploadFile = File(...)):
    """
    Extract text from uploaded image using optimized OCR pipeline.
    
    Includes explicit memory cleanup (del, gc.collect) after processing 
    to ensure stability on low-RAM containers.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # Run optimized preprocessing
        processed_img = advanced_preprocess(nparr)
        
        # Tesseract Configuration:
        # --psm 6: Assume a single uniform block of text (Best for tables)
        custom_config = r'--oem 1 --psm 6' 
        raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        clean_text = clean_text_for_llm(raw_text)
        
        # CRITICAL: Free memory immediately
        del processed_img
        del nparr
        gc.collect()
        
        return {"status": "success", "text": clean_text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/debug-view")
async def debug_view(file: UploadFile = File(...)):
    """
    Return processed image for debugging.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    processed_img = advanced_preprocess(nparr)
    _, encoded_img = cv2.imencode('.jpg', processed_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@app.get("/")
def home():
    return {"message": "✅ OCR Backend Service - v8.0 (Low Memory Optimized)"}