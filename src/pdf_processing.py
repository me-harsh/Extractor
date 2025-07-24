import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
import cv2
import os
from datetime import datetime

def load_pdf_with_fitz(pdf_path):
    """Load PDF using PyMuPDF (fitz)"""
    try:
        doc = fitz.open(pdf_path)
        print(f"Loaded PDF with {len(doc)} pages")
        return doc
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

def pdf_page_to_image(page, dpi=300):
    """Convert PDF page to image using fitz"""
    try:
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        return img, pix
    except Exception as e:
        print(f"Error converting page to image: {e}")
        return None, None

def extract_text_with_fitz(page):
    """Extract text directly using fitz"""
    try:
        text = page.get_text()
        if text.strip() and len(text.strip()) > 50:
            print("Using direct text extraction (faster)")
            return text, "direct"
        else:
            print("Direct text extraction yielded little text, will use OCR")
            return "", "needs_ocr"
    except Exception as e:
        print(f"Error in direct text extraction: {e}")
        return "", "needs_ocr"
    



