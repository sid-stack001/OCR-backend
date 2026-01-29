"""
OCR Backend Service - FastAPI Application (v9.0 Otsu Edition)

This version switches to Otsu's Binarization to solve the "Hollow Bold Text" 
problem found in previous versions. It is robust for bold medical headers.

Author: Siddharth Verma
Version: 3.0
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
import gc

app = FastAPI()

# Initialize background removal model
session = new_session("u2netp")

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def remove_vertical_lines(image):
    # Kernel: 1px wide, 40px tall
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
    result = cv2.add(image, detected_lines)
    return result

def clean_text_for_llm(raw_text):
    lines = raw_text.split('\n')
    cleaned_lines = []
    for line in lines:
        clean = line.replace('|', ' ').replace('_', ' ').replace('—', ' ')
        clean = re.sub(r'\[\s*\]', '', clean)
        clean = re.sub(r'\(\s*\)', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if len(clean) > 2:
            cleaned_lines.append(clean)
    return "\n".join(cleaned_lines)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
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
# IMAGE PREPROCESSING PIPELINE (v9.0 - Otsu)
# ============================================================================

def advanced_preprocess(img_array):
    original = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if original is None: 
        raise ValueError("Image decoding failed")

    # --- STAGE 1: Memory-Optimized AI Detection ---
    scale_factor = 1.0
    h, w = original.shape[:2]
    
    # 1024px limit for AI mask generation (Saves RAM)
    if w > 1024:
        scale_factor = 1024 / w
        input_for_ai = cv2.resize(original, (1024, int(h * scale_factor)))
    else:
        input_for_ai = original

    warped = original
    
    try:
        pil_img = Image.fromarray(cv2.cvtColor(input_for_ai, cv2.COLOR_BGR2RGB))
        no_bg = remove(pil_img, session=session)
        
        del pil_img
        gc.collect()

        alpha = np.array(no_bg)[:, :, 3]
        cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > (0.10 * (input_for_ai.shape[0]*input_for_ai.shape[1])):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                if scale_factor != 1.0:
                    approx = (approx / scale_factor).astype(np.float32)

                if len(approx) == 4:
                    warped = four_point_transform(original, approx.reshape(4, 2))
                else:
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    if scale_factor != 1.0:
                        box = (box / scale_factor).astype(np.float32)
                    box = np.int32(box)
                    warped = four_point_transform(original, box)
    except Exception:
        warped = original

    gc.collect()

    # --- STAGE 2: Gamma Correction ---
    # 0.5 makes text "thicker" / bolder
    warped = adjust_gamma(warped, gamma=0.5)

    # --- STAGE 3: Upscaling (2000px) ---
    # We push back to 2000px for better definition
    target_width = 2000
    scale = target_width / warped.shape[1]
    if scale > 1:
        dim = (target_width, int(warped.shape[0] * scale))
        warped = cv2.resize(warped, dim, interpolation=cv2.INTER_CUBIC)

    # --- STAGE 4: Denoising ---
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur is REQUIRED for Otsu to work well
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- STAGE 5: Otsu's Binarization (THE FIX) ---
    # Replaces Adaptive Threshold. 
    # Global thresholding keeps bold text solid, preventing "hollow" letters.
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        processed_img = advanced_preprocess(nparr)
        
        custom_config = r'--oem 1 --psm 6' 
        raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        clean_text = clean_text_for_llm(raw_text)
        
        # Cleanup
        del processed_img
        del nparr
        gc.collect()
        
        return {"status": "success", "text": clean_text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/debug-view")
async def debug_view(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    processed_img = advanced_preprocess(nparr)
    _, encoded_img = cv2.imencode('.jpg', processed_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.get("/")
def home():
    return {"message": "✅ OCR Backend Service - v3.0 (Otsu Edition)"}
