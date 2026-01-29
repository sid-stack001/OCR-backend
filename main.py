"""
OCR Backend Service - FastAPI Application

This module implements an advanced OCR pipeline with background removal,
perspective correction, image preprocessing, and text extraction using
Tesseract OCR. The service provides endpoints for document digitization
and debugging of the preprocessing pipeline.

Author: Siddharth Verma
Version: 7.0
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

app = FastAPI()

# Initialize background removal model
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
        gamma (float): Gamma value. <1.0 darkens, >1.0 lightens. Default: 1.0
    
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
    
    Uses morphological operations with a tall, thin kernel to detect and
    remove only vertical lines. Horizontal lines are preserved to maintain
    text underlines and other horizontal elements.
    
    Args:
        image (np.ndarray): Binary image (text on white background)
    
    Returns:
        np.ndarray: Image with vertical lines removed
    """
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
    result = cv2.add(image, detected_lines)
    return result


def clean_text_for_llm(raw_text):
    """
    Clean OCR output text for downstream LLM processing.
    
    Removes special characters, empty brackets/parentheses, and normalizes
    whitespace. Filters out lines with minimal content.
    
    Args:
        raw_text (str): Raw OCR output from Tesseract
    
    Returns:
        str: Cleaned text ready for LLM processing
    """
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
    """
    Order corner points in consistent sequence: top-left, top-right, bottom-right, bottom-left.
    
    Used as preprocessing for perspective transformation.
    
    Args:
        pts (np.ndarray): Array of 4 corner points
    
    Returns:
        np.ndarray: Reordered points in standard sequence
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
    Apply perspective transformation using four corner points.
    
    Warps the image so that the region defined by the four points appears
    as a rectangular document viewed straight-on.
    
    Args:
        image (np.ndarray): Input image
        pts (np.ndarray): Four corner points defining the region
    
    Returns:
        np.ndarray: Perspective-corrected image
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
# IMAGE PREPROCESSING PIPELINE
# ============================================================================

def advanced_preprocess(img_array):
    """
    Execute comprehensive image preprocessing pipeline for OCR.
    
    Pipeline stages:
    1. AI-based document detection and perspective correction
    2. Gamma adjustment for optimal text visibility
    3. Image upscaling for improved character recognition
    4. Bilateral filtering to reduce noise while preserving edges
    5. Adaptive thresholding for text/background separation
    6. Vertical line removal to uncage table text
    7. Border padding for OCR engine compatibility
    
    Args:
        img_array (np.ndarray): Raw image bytes as numpy array
    
    Returns:
        np.ndarray: Processed binary image ready for OCR
    
    Raises:
        ValueError: If image decoding fails
    """
    original = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if original is None: 
        raise ValueError("Image decoding failed")

    # Stage 1: AI-based document detection and perspective correction
    try:
        pil_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        no_bg = remove(pil_img, session=session)
        alpha = np.array(no_bg)[:, :, 3]
        cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        warped = original
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > (0.10 * (original.shape[0]*original.shape[1])):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    warped = four_point_transform(original, approx.reshape(4, 2))
                else:
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    warped = four_point_transform(original, box)
    except:
        warped = original

    # Stage 2: Gamma correction (0.5 brightens faint text)
    warped = adjust_gamma(warped, gamma=0.5)

    # Stage 3: Upscaling to 3000px width for improved character spacing and recognition
    target_width = 3000
    scale = target_width / warped.shape[1]
    if scale > 1:
        dim = (target_width, int(warped.shape[0] * scale))
        warped = cv2.resize(warped, dim, interpolation=cv2.INTER_CUBIC)

    # Stage 4: Convert to grayscale and apply bilateral filtering
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Stage 5: Adaptive thresholding for robust text/background separation
    # Block size: 61 (prevents bold text from becoming hollow)
    # Constant: 15 (maintains clean background)
    binary = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        61, 15 
    )

    # Stage 6: Remove vertical lines while preserving text integrity
    clean = remove_vertical_lines(binary)

    # Stage 7: Add border padding for Tesseract OCR compatibility
    clean = cv2.copyMakeBorder(clean, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return clean



# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/ocr")
async def get_ocr(file: UploadFile = File(...)):
    """
    Extract text from uploaded image using OCR pipeline.
    
    Processes the image through the complete preprocessing pipeline,
    then uses Tesseract with PSM 6 (layout analysis) for text extraction.
    Output is cleaned and normalized for downstream processing.
    
    Args:
        file (UploadFile): Image file (PNG, JPG, etc.)
    
    Returns:
        dict: JSON response with status and extracted text or error message
            {
                "status": "success" or "error",
                "text": "extracted text",
                "message": "error details if status is error"
            }
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        processed_img = advanced_preprocess(nparr)
        
        # Tesseract configuration:
        # --oem 1: Use legacy OCR engine mode
        # --psm 6: Layout analysis (suitable for structured documents and tables)
        custom_config = r'--oem 1 --psm 6' 
        raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        clean_text = clean_text_for_llm(raw_text)
        
        return {"status": "success", "text": clean_text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/debug-view")
async def debug_view(file: UploadFile = File(...)):
    """
    Return preprocessed image for visual debugging and pipeline validation.
    
    Applies the complete preprocessing pipeline and returns the resulting
    binary image as JPEG. Useful for troubleshooting OCR quality issues.
    
    Args:
        file (UploadFile): Image file (PNG, JPG, etc.)
    
    Returns:
        Response: Binary JPEG image showing preprocessing results
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    processed_img = advanced_preprocess(nparr)
    _, encoded_img = cv2.imencode('.jpg', processed_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@app.get("/")
def home():
    """
    Health check endpoint.
    
    Returns:
        dict: Service status message
    """
    return {"message": "OCR Backend Service - v7.0 (Ready)"}