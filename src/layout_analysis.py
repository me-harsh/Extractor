import layoutparser as lp
import cv2
import numpy as np
from PIL import Image

def detect_layout_ocr(image):
    """Use layout detection + OCR for scanned PDFs"""
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_np
    
    try:
        model = lp.Detectron2LayoutModel(
            "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
        layout = model.detect(image_rgb)
        return layout, image_rgb
    except Exception as e:
        print(f"Layout detection error: {e}")
        return None, image_rgb

def extract_text_with_ocr(layout, image):
    """Extract text using OCR when direct extraction fails"""
    ocr_agent = lp.TesseractAgent(languages='eng')
    text_blocks = [block for block in layout if block.type == "Text"]
    text_blocks.sort(key=lambda b: (b.block.y1, b.block.x1))
    
    full_text = ""
    for block in text_blocks:
        try:
            padded_block = block.pad(left=3, right=3, top=3, bottom=3)
            cropped_image = padded_block.crop_image(image)
            text = ocr_agent.detect(cropped_image)
            if text.strip():
                full_text += text.strip() + "\n"
        except Exception as e:
            print(f"OCR error: {e}")
            continue
    
    return full_text