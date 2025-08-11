import fitz  # PyMuPDF
from openai import OpenAI
import base64
import requests
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageStat
from typing import List, Dict, Union, Optional
import os
from dotenv import load_dotenv
from pathlib import Path
import numpy as np

# === CONFIG ===
# IMAGE_VLLM_URL = "http://localhost:9019/v1"
# IMAGE_VLLM_MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

# Default settings - can be overridden
class Settings:
    IMAGE_MIN_SIZE = 70
    IMAGE_MAX_SIZE = 2000
    IMAGE_OVERLAP_THRESHOLD = 20

settings = Settings()

# ======================
# OpenAI Client Setup
# ======================

client = OpenAI(
    # api_key=os.getenv("OPENAI_API_KEY"),
    # base_url=os.getenv("IMAGE_VLLM_URL") 
    api_key="ahosi", #remove this line
    base_url="http://localhost:8011/v1" #remove this line
)

# ======================
# Advanced Image Filtering Functions (from actual.py)
# ======================

def _is_image_in_visible_area(page, xref: int, img_num: int) -> bool:
    """
    Check if an image is actually within the visible area of the partitioned PDF page.
    This is the key method to prevent analyzing images from original PDF that are outside the cropped area.
    """
    try:
        # Method 1: Check image placement using page annotations and image objects
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # Method 2: Use get_image_rects to get actual placement rectangles
        try:
            image_rects = page.get_image_rects(xref)
            if not image_rects:
                print(f"🔍 DEBUG: Image {img_num}: No placement rectangles found")
                return False
            
            print(f"🔍 DEBUG: Image {img_num}: Found {len(image_rects)} placement rectangles")
            
            # Check if any rectangle is within the visible page area
            for rect in image_rects:
                rect_x0, rect_y0, rect_x1, rect_y1 = rect
                
                print(f"🔍 DEBUG: Image {img_num}: Rect bounds: ({rect_x0:.1f}, {rect_y0:.1f}) to ({rect_x1:.1f}, {rect_y1:.1f})")
                print(f"🔍 DEBUG: Image {img_num}: Page bounds: (0, 0) to ({page_width:.1f}, {page_height:.1f})")
                
                # Check if rectangle overlaps with visible page area
                # An image is visible if it has any overlap with the page bounds
                if (rect_x1 > 0 and rect_x0 < page_width and 
                    rect_y1 > 0 and rect_y0 < page_height):
                    
                    # Calculate overlap area to determine if it's meaningful
                    overlap_x0 = max(0, rect_x0)
                    overlap_y0 = max(0, rect_y0)
                    overlap_x1 = min(page_width, rect_x1)
                    overlap_y1 = min(page_height, rect_y1)
                    
                    overlap_width = max(0, overlap_x1 - overlap_x0)
                    overlap_height = max(0, overlap_y1 - overlap_y0)
                    overlap_area = overlap_width * overlap_height
                    
                    # Calculate original rectangle area
                    rect_width = rect_x1 - rect_x0
                    rect_height = rect_y1 - rect_y0
                    rect_area = rect_width * rect_height
                    
                    if rect_area > 0:
                        overlap_percentage = (overlap_area / rect_area) * 100
                        print(f"🔍 DEBUG: Image {img_num}: Overlap: {overlap_percentage:.1f}% ({overlap_area:.1f}/{rect_area:.1f})")
                        
                        # Only consider images with significant overlap
                        # This prevents tiny edge overlaps from being processed
                        min_overlap_threshold = getattr(settings, 'IMAGE_OVERLAP_THRESHOLD', 20)  # Default 20%
                        if overlap_percentage > min_overlap_threshold:
                            print(f"🔍 DEBUG: Image {img_num}: ✅ VISIBLE - {overlap_percentage:.1f}% overlap")
                            return True
                        else:
                            print(f"🔍 DEBUG: Image {img_num}: ❌ MINIMAL OVERLAP - {overlap_percentage:.1f}%")
                    else:
                        print(f"🔍 DEBUG: Image {img_num}: ❌ ZERO AREA")
                else:
                    print(f"🔍 DEBUG: Image {img_num}: ❌ NO OVERLAP with page bounds")
            
            print(f"🔍 DEBUG: Image {img_num}: ❌ NOT VISIBLE - no meaningful overlaps")
            return False
            
        except Exception as rect_error:
            print(f"🔍 DEBUG: Image {img_num}: get_image_rects failed: {rect_error}")
            
            # Fallback Method 3: Render-based detection
            # If we can't get rectangles, try rendering the page and checking if image appears
            return _fallback_render_check(page, xref, img_num)
            
    except Exception as e:
        print(f"🔍 DEBUG: Image {img_num}: Visibility check failed: {e}")
        # If we can't determine, err on the side of inclusion
        return True

def _fallback_render_check(page, xref: int, img_num: int) -> bool:
    """
    Fallback method: Render the page and check if the image actually appears.
    This is more resource-intensive but very reliable.
    """
    try:
        print(f"🔍 DEBUG: Image {img_num}: Using fallback render check")
        
        # Render the page at low resolution for efficiency
        mat = fitz.Matrix(0.5, 0.5)  # 50% scale for speed
        pixmap = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image for analysis
        img_data = pixmap.tobytes("png")
        rendered_page = Image.open(BytesIO(img_data))
        
        # Get the original image
        try:
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            original_image = Image.open(BytesIO(image_bytes))
            
            # Resize original image to match rendered scale
            original_resized = original_image.resize(
                (int(original_image.width * 0.5), int(original_image.height * 0.5)), 
                Image.Resampling.LANCZOS
            )
            
            # Simple correlation check - if the image appears in the rendered page
            # This is a simplified check - in practice, you might want more sophisticated matching
            rendered_array = np.array(rendered_page.convert('L'))
            original_array = np.array(original_resized.convert('L'))
            
            # If original image is small enough to fit in rendered page, it might be visible
            if (original_array.shape[0] <= rendered_array.shape[0] and 
                original_array.shape[1] <= rendered_array.shape[1]):
                print(f"🔍 DEBUG: Image {img_num}: ✅ LIKELY VISIBLE - size compatible")
                return True
            else:
                print(f"🔍 DEBUG: Image {img_num}: ❌ TOO LARGE - likely outside bounds")
                return False
                
        except Exception as extract_error:
            print(f"🔍 DEBUG: Image {img_num}: Render check extraction failed: {extract_error}")
            return True  # Default to visible if we can't determine
            
    except Exception as e:
        print(f"🔍 DEBUG: Image {img_num}: Fallback render check failed: {e}")
        return True  # Default to visible if we can't determine

def _extract_image_with_fallbacks(page, xref: int, img_num: int) -> Optional[dict]:
    """Extract image using multiple fallback methods to handle various PDF encodings"""
    
    # Method 1: Standard extraction
    try:
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(BytesIO(image_bytes))
        
        # Check if extraction seems successful (not corrupted)
        if image.size[0] > 0 and image.size[1] > 0:
            print(f"🔍 DEBUG: Image {img_num}: Method 1 (standard) successful")
            return {'image': image, 'method': 'standard_extraction', 'colorspace': base_image.get('colorspace', 'unknown')}
    except Exception as e:
        print(f"🔍 DEBUG: Image {img_num}: Method 1 failed - {e}")
    
    # Method 2: Extract without alpha
    try:
        base_image = page.parent.extract_image(xref, keep_alpha=False)
        image_bytes = base_image["image"]
        image = Image.open(BytesIO(image_bytes))
        
        if image.size[0] > 0 and image.size[1] > 0:
            print(f"🔍 DEBUG: Image {img_num}: Method 2 (no alpha) successful")
            return {'image': image, 'method': 'no_alpha_extraction', 'colorspace': base_image.get('colorspace', 'unknown')}
    except Exception as e:
        print(f"🔍 DEBUG: Image {img_num}: Method 2 failed - {e}")
    
    # Method 3: Use pixmap rendering (rasterization)
    try:
        # Get image bbox from the page
        img_info = [img for img in page.get_images(full=True) if img[0] == xref][0]
        
        # Create a small pixmap to render just this image area
        # This method rasterizes the image, which can fix color space issues
        mat = fitz.Matrix(2.0, 2.0)  # 2x scale for better quality
        pixmap = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert pixmap to PIL Image
        img_data = pixmap.tobytes("png")
        image = Image.open(BytesIO(img_data))
        
        if image.size[0] > 0 and image.size[1] > 0:
            print(f"🔍 DEBUG: Image {img_num}: Method 3 (pixmap) successful")
            return {'image': image, 'method': 'pixmap_rendering', 'colorspace': 'rendered'}
    except Exception as e:
        print(f"🔍 DEBUG: Image {img_num}: Method 3 failed - {e}")
    
    # Method 4: Try with different color space conversion
    try:
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        
        # Try opening with different PIL parameters
        image = Image.open(BytesIO(image_bytes))
        
        # Force convert through different color spaces to fix encoding issues
        if image.mode == 'CMYK':
            # CMYK often causes inversion issues
            image = image.convert('RGB')
        elif image.mode not in ['RGB', 'RGBA', 'L']:
            image = image.convert('RGB')
        
        if image.size[0] > 0 and image.size[1] > 0:
            print(f"🔍 DEBUG: Image {img_num}: Method 4 (color conversion) successful")
            return {'image': image, 'method': 'color_space_conversion', 'colorspace': base_image.get('colorspace', 'unknown')}
    except Exception as e:
        print(f"🔍 DEBUG: Image {img_num}: Method 4 failed - {e}")
    
    print(f"🔍 DEBUG: Image {img_num}: All extraction methods failed")
    return None

def _apply_color_corrections(image: Image.Image, img_num: int) -> Image.Image:
    """Apply enhanced color corrections to fix common PDF extraction issues"""
    
    try:
        original_mode = image.mode
        print(f"🔍 DEBUG: Image {img_num}: Applying color corrections (original mode: {original_mode})")
        
        # Convert to RGB for processing
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Create white background and composite
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode in ['L', 'P', 'CMYK']:
                image = image.convert('RGB')
        
        # Enhanced inversion detection using multiple methods
        inversion_confidence = _detect_color_inversion_confidence(image)
        
        print(f"🔍 DEBUG: Image {img_num}: Inversion confidence: {inversion_confidence:.2f}")
        
        # Apply inversion if confidence is high
        if inversion_confidence > 0.7:  # Threshold for applying inversion
            print(f"🔍 DEBUG: Image {img_num}: Applying color inversion (confidence: {inversion_confidence:.2f})")
            image = ImageOps.invert(image)
            print(f"🎨 Image {img_num}: Applied color inversion (confidence: {inversion_confidence:.2f})")
        
        # Additional color enhancements
        image = _enhance_image_quality(image, img_num)
        
        return image
        
    except Exception as e:
        print(f"⚠️ Image {img_num}: Color correction failed - {e}")
        return image

def _detect_color_inversion_confidence(image: Image.Image) -> float:
    """Calculate confidence score for whether an image needs color inversion"""
    
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Multiple detection methods
        confidence_scores = []
        
        # Method 1: Overall brightness analysis
        mean_brightness = np.mean(img_array)
        brightness_confidence = max(0, (50 - mean_brightness) / 50)  # Higher confidence if very dark
        confidence_scores.append(brightness_confidence * 0.4)
        
        # Method 2: Edge analysis (edges should be dark in normal images)
        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_brightness = np.mean(np.array(edges))
        edge_confidence = min(1.0, edge_brightness / 100)  # Higher confidence if edges are bright
        confidence_scores.append(edge_confidence * 0.3)
        
        # Method 3: Background analysis (corners should be light in normal images)
        h, w = img_array.shape[:2]
        corner_regions = [
            img_array[0:h//4, 0:w//4],          # Top-left
            img_array[0:h//4, 3*w//4:w],        # Top-right
            img_array[3*h//4:h, 0:w//4],        # Bottom-left
            img_array[3*h//4:h, 3*w//4:w]       # Bottom-right
        ]
        
        corner_brightness = np.mean([np.mean(region) for region in corner_regions])
        background_confidence = max(0, (80 - corner_brightness) / 80)
        confidence_scores.append(background_confidence * 0.3)
        
        # Combined confidence
        total_confidence = sum(confidence_scores)
        
        return min(1.0, total_confidence)
        
    except Exception as e:
        print(f"⚠️ Color inversion detection failed: {e}")
        return 0.0

def _enhance_image_quality(image: Image.Image, img_num: int) -> Image.Image:
    """Apply subtle quality enhancements"""
    
    try:
        # Only apply enhancements if image is not too small
        if min(image.size) < 100:
            return image
        
        # Subtle contrast enhancement
        # Check if image needs contrast adjustment
        stat = ImageStat.Stat(image)
        
        # If image has very low contrast, enhance it slightly
        if len(stat.stddev) == 3:  # RGB
            avg_stddev = sum(stat.stddev) / 3
            if avg_stddev < 20:  # Low contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)  # Subtle enhancement
                print(f"🔍 DEBUG: Image {img_num}: Applied contrast enhancement")
        
        return image
        
    except Exception as e:
        print(f"⚠️ Image {img_num}: Quality enhancement failed - {e}")
        return image

def _validate_image_comprehensively(image: Image.Image, img_num: int) -> dict:
    """Comprehensive image validation with multiple checks"""
    
    width, height = image.size
    min_size = getattr(settings, 'IMAGE_MIN_SIZE', 70)
    max_size = getattr(settings, 'IMAGE_MAX_SIZE', 2000)  # Increased from 800
    
    # Size validation
    if width < min_size or height < min_size:
        return {'is_valid': False, 'reason': f'TOO_SMALL - {width}x{height}px < {min_size}px'}
    
    if width > max_size or height > max_size:
        return {'is_valid': False, 'reason': f'TOO_LARGE - {width}x{height}px > {max_size}px'}
    
    # Format validation
    if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
        return {'is_valid': False, 'reason': f'UNSUPPORTED_FORMAT - {image.mode}'}
    
    # Enhanced decorative detection
    decorative_check = _is_decorative_image_enhanced(image)
    if decorative_check['is_decorative']:
        return {'is_valid': False, 'reason': f'DECORATIVE - {decorative_check["reason"]}'}
    
    # Content richness check
    richness_check = _check_content_richness(image)
    if not richness_check['has_content']:
        return {'is_valid': False, 'reason': f'LOW_CONTENT - {richness_check["reason"]}'}
    
    return {'is_valid': True, 'reason': 'PASSED_ALL_CHECKS'}

def _is_decorative_image_enhanced(image: Image.Image) -> dict:
    """Enhanced decorative image detection with special handling for timing diagrams and technical content"""
    
    try:
        # Convert to RGB for consistent analysis
        if image.mode != 'RGB':
            analysis_image = image.convert('RGB')
        else:
            analysis_image = image
        
        width, height = analysis_image.size
        
        # Get image statistics
        stat = ImageStat.Stat(analysis_image)
        
        # SPECIAL CASE: Check if this looks like a timing diagram or technical diagram
        # These often have lots of white space but are very important
        if _is_likely_timing_or_technical_diagram(analysis_image):
            print(f"🔍 DEBUG: Detected likely timing/technical diagram - forcing inclusion")
            return {'is_decorative': False, 'reason': 'timing_or_technical_diagram_detected'}
        
        # Method 1: Single color dominance (more relaxed for technical content)
        pixels = list(analysis_image.getdata())
        color_counts = {}
        white_pixels = 0
        black_pixels = 0
        
        for pixel in pixels:
            color_counts[pixel] = color_counts.get(pixel, 0) + 1
            # Count white pixels (very light)
            if all(val >= 250 for val in pixel):
                white_pixels += 1
            # Count black pixels (very dark)
            elif all(val <= 5 for val in pixel):
                black_pixels += 1
        
        max_color_count = max(color_counts.values())
        single_color_ratio = max_color_count / len(pixels)
        
        # More strict threshold for single color (99.5% instead of 98%)
        if single_color_ratio > 0.995:
            return {'is_decorative': True, 'reason': f'single_color_dominant_{single_color_ratio:.3f}'}
        
        # Method 2: Pure white dominance (very strict - 97% instead of 95%)
        light_ratio = white_pixels / len(pixels)
        if light_ratio > 0.97:
            return {'is_decorative': True, 'reason': f'mostly_white_{light_ratio:.3f}'}
        
        # Method 3: Check for meaningful line structures
        # Images with horizontal/vertical lines are often technical diagrams
        if _has_meaningful_line_structure(analysis_image):
            print(f"🔍 DEBUG: Detected meaningful line structure - likely diagram")
            return {'is_decorative': False, 'reason': 'has_line_structure'}
        
        # Method 4: Very low variance (almost no variation) - but be more lenient
        if len(stat.stddev) == 3:  # RGB
            avg_stddev = sum(stat.stddev) / 3
            if avg_stddev < 2:  # Very very low variation (stricter threshold)
                return {'is_decorative': True, 'reason': f'extremely_low_variation_{avg_stddev:.1f}'}
        
        # Method 5: Aspect ratio check for lines/borders (more lenient)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 30:  # Very thin lines (was 20, now 30)
            return {'is_decorative': True, 'reason': f'very_thin_line_{aspect_ratio:.1f}'}
        
        # Method 6: Check if it's just a simple border or frame
        if _is_simple_border_or_frame(analysis_image):
            return {'is_decorative': True, 'reason': 'simple_border_or_frame'}
        
        return {'is_decorative': False, 'reason': 'has_meaningful_content'}
        
    except Exception as e:
        print(f"⚠️ Enhanced decorative check failed: {e}")
        return {'is_decorative': False, 'reason': 'check_failed'}

def _is_likely_timing_or_technical_diagram(image: Image.Image) -> bool:
    """Detect if image is likely a timing diagram, circuit diagram, or other technical diagram"""
    
    try:
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Check for horizontal lines (common in timing diagrams)
        horizontal_lines = 0
        for y in range(height):
            row = gray[y, :]
            # Look for transitions from light to dark and back (indicating signals)
            transitions = np.sum(np.abs(np.diff(row)) > 50)
            if transitions >= 2:  # At least 2 transitions in a row
                horizontal_lines += 1
        
        horizontal_line_ratio = horizontal_lines / height
        
        # Check for vertical lines (time markers, grid lines)
        vertical_lines = 0
        for x in range(width):
            col = gray[:, x]
            transitions = np.sum(np.abs(np.diff(col)) > 50)
            if transitions >= 1:  # At least 1 transition in a column
                vertical_lines += 1
        
        vertical_line_ratio = vertical_lines / width
        
        # Timing diagrams typically have:
        # - Multiple horizontal lines (signal traces)
        # - Some vertical lines (time markers)
        # - Rectangular patterns (digital signals)
        
        print(f"🔍 DEBUG: Line analysis - horizontal: {horizontal_line_ratio:.3f}, vertical: {vertical_line_ratio:.3f}")
        
        # If we have reasonable horizontal and some vertical line structure
        if horizontal_line_ratio > 0.1 and vertical_line_ratio > 0.05:
            return True
        
        # Check for text labels (common in timing diagrams)
        # Look for regions with medium gray values (text)
        text_pixels = np.sum((gray > 50) & (gray < 200))
        text_ratio = text_pixels / (height * width)
        
        # If we have some text and line structure, likely a technical diagram
        if text_ratio > 0.05 and (horizontal_line_ratio > 0.05 or vertical_line_ratio > 0.03):
            return True
        
        return False
        
    except Exception as e:
        print(f"🔍 DEBUG: Timing diagram detection failed: {e}")
        return False

def _has_meaningful_line_structure(image: Image.Image) -> bool:
    """Check if image has meaningful line structures (rectangles, grids, etc.)"""
    
    try:
        # Convert to grayscale
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        
        img_array = np.array(gray_image)
        height, width = img_array.shape
        
        # Look for rectangular patterns (common in timing diagrams)
        rectangles_detected = 0
        
        # Simple rectangle detection: look for regions with consistent borders
        for y in range(0, height-10, 10):  # Sample every 10 pixels
            for x in range(0, width-10, 10):
                # Check small region for rectangular pattern
                region = img_array[y:y+10, x:x+10]
                if region.shape[0] == 10 and region.shape[1] == 10:
                    # Check if borders are different from interior
                    border_mean = np.mean([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
                    interior_mean = np.mean(region[2:-2, 2:-2])
                    
                    if abs(border_mean - interior_mean) > 30:  # Significant difference
                        rectangles_detected += 1
        
        # If we found several rectangular patterns, likely meaningful
        total_regions = (height // 10) * (width // 10)
        rectangle_ratio = rectangles_detected / max(1, total_regions)
        
        print(f"🔍 DEBUG: Rectangle analysis - ratio: {rectangle_ratio:.3f}")
        
        return rectangle_ratio > 0.02  # At least 2% of regions show rectangular patterns
        
    except Exception as e:
        print(f"🔍 DEBUG: Line structure detection failed: {e}")
        return False

def _is_simple_border_or_frame(image: Image.Image) -> bool:
    """Check if image is just a simple border or frame with no content"""
    
    try:
        # Convert to grayscale
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        
        img_array = np.array(gray_image)
        height, width = img_array.shape
        
        # Check if interior is mostly uniform while borders are different
        border_width = min(5, width // 10, height // 10)  # Adaptive border width
        
        if border_width < 1:
            return False
        
        # Extract border and interior regions
        top_border = img_array[:border_width, :]
        bottom_border = img_array[-border_width:, :]
        left_border = img_array[:, :border_width]
        right_border = img_array[:, -border_width:]
        
        interior = img_array[border_width:-border_width, border_width:-border_width]
        
        if interior.size == 0:
            return True  # Too small, likely just a border
        
        # Calculate variations
        border_std = np.std([top_border, bottom_border, left_border, right_border])
        interior_std = np.std(interior)
        
        # If interior is very uniform but borders have some variation, likely just a frame
        if interior_std < 10 and border_std > 20:
            return True
        
        return False
        
    except Exception as e:
        print(f"🔍 DEBUG: Border detection failed: {e}")
        return False

def _check_content_richness(image: Image.Image) -> dict:
    """Check if image has rich enough content to be meaningful"""
    
    try:
        # Convert to RGB for analysis
        if image.mode != 'RGB':
            analysis_image = image.convert('RGB')
        else:
            analysis_image = image
        
        # Get basic statistics
        stat = ImageStat.Stat(analysis_image)
        
        # Check color diversity
        unique_colors = len(set(analysis_image.getdata()))
        total_pixels = analysis_image.size[0] * analysis_image.size[1]
        color_diversity = unique_colors / total_pixels
        
        # Very low color diversity might indicate simple graphics
        if color_diversity < 0.01 and unique_colors < 10:
            return {'has_content': False, 'reason': f'low_color_diversity_{color_diversity:.4f}'}
        
        # Check if it's likely text or meaningful content
        # Text usually has good contrast and moderate complexity
        if len(stat.stddev) == 3:  # RGB
            avg_stddev = sum(stat.stddev) / 3
            
            # Images with some variation are more likely to be meaningful
            if avg_stddev > 8:  # Has some variation
                return {'has_content': True, 'reason': f'good_variation_{avg_stddev:.1f}'}
        
        # If we reach here, it's borderline - err on the side of inclusion
        return {'has_content': True, 'reason': 'borderline_included'}
        
    except Exception as e:
        print(f"⚠️ Content richness check failed: {e}")
        return {'has_content': True, 'reason': 'check_failed_included'}

def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Save as PNG for best quality
    image.save(buffered, format="PNG", optimize=True)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_base64

def _is_llm_identified_decorative(description: str) -> bool:
    """Check if LLM identified the image as decorative"""
    if not description:
        return True
    
    decorative_indicators = [
        'DECORATIVE_ELEMENT',
        'decorative element',
        'purely decorative',
        'no educational value',
        'logo',
        'watermark'
    ]
    
    description_lower = description.lower()
    return any(indicator in description_lower for indicator in decorative_indicators)

# ======================
# Enhanced Image Description function (replacing the simple relevance check)
# ======================

def describe_educational_image_with_base64(base64_image):
    """Describe an image given its base64 encoding using the correct LLM structure."""
    print("Getting image description...")
    response = client.chat.completions.create(
        # model=os.getenv("IMAGE_VLLM_MODEL_NAME"),
        model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",#remove this line
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at analyzing educational content images. "
                    "Provide clear, structured descriptions that capture essential educational elements. "
                    "Focus on mathematical formulas, diagrams, charts, graphs, illustrations, and text content. "
                    "If the image is clearly a logo, watermark, header/footer, or purely decorative element with no educational value, "
                    "respond with exactly 'DECORATIVE_ELEMENT'. "
                    "For meaningful content, be concise but comprehensive."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image and provide a structured description focusing on educational content:\n"
                            "- Content type: [mathematical formula/diagram/chart/graph/illustration/text/etc.]\n"
                            "- Key elements: [main components, labels, values]\n"
                            "- Educational purpose: [what concept it teaches or demonstrates]\n"
                            "- Important details: [formulas, relationships, data points, include object positions, shapes, sizes, label names, arrows, styles, fonts, spacing, and any implied relationships.]\n\n"
                            "Keep it concise and relevant to learning. "
                            "Only respond with 'DECORATIVE_ELEMENT' if this has absolutely no educational value."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=2000
    )
    
    return response.choices[0].message.content

# ======================
# Image Saving function (unchanged)
# ======================

def save_image_to_folder(image_bytes, pdf_name, img_index, output_dir):
    """Save image to the images folder with proper naming convention."""
    try:
        # Create images folder inside the output directory
        images_folder = os.path.join(output_dir, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # Clean PDF name (remove extension and any path components)
        clean_pdf_name = os.path.splitext(os.path.basename(pdf_name))[0]
        
        # Create filename: pdf_name_image_1.png, pdf_name_image_2.png, etc.
        image_filename = f"{clean_pdf_name}_image_{img_index + 1}.png"
        image_path = os.path.join(images_folder, image_filename)
        
        # Convert to PIL Image and save
        image = Image.open(BytesIO(image_bytes))
        image.save(image_path, format="PNG")
        
        print(f"Saved image: {image_path}")
        return image_path
        
    except Exception as e:
        print(f"Error saving image {img_index}: {str(e)}")
        return None

# ======================
# MAIN ENHANCED IMAGE EXTRACTOR (replacing extract_image_descriptions_from_page)
# ======================

def extract_image_descriptions_from_page(page, pdf_name=None, output_dir=None):
    """
    Enhanced image extraction with comprehensive filtering from actual.py
    Extracts images from the page, applies advanced filtering, saves relevant ones, and gets descriptions.
    """
    print("Inside extract_image_description_from_page (ENHANCED VERSION)")
    images = list(page.get_images(full=True))
    total_images = len(images)
    image_descriptions = []
    relevant_image_count = 0
    
    print(f"🔍 DEBUG: extract_image_descriptions_from_page called with {total_images} images")
    
    # Get page dimensions to understand the actual content area
    page_rect = page.rect
    page_width = page_rect.width
    page_height = page_rect.height
    
    print(f"🔍 DEBUG: Page dimensions: {page_width}x{page_height}")
    
    # Only save images if output_dir is provided
    if output_dir is not None:
        save_images = True
        if pdf_name is None:
            pdf_name = "unknown_pdf"  # fallback only if output_dir is provided
    else:
        save_images = False
    
    for img_index, img in enumerate(images):
        print(f"🔍 DEBUG: Processing image {img_index + 1}/{total_images}")
        
        try:
            xref = img[0]
            
            # STEP 1: Check if image is actually within the visible page area
            # This is the key fix for partitioned PDFs
            if not _is_image_in_visible_area(page, xref, img_index + 1):
                print(f"🔍 DEBUG: Image {img_index + 1}: Skipped - outside visible area")
                continue
            
            # STEP 2: Enhanced image extraction with multiple fallback methods
            image_data = _extract_image_with_fallbacks(page, xref, img_index + 1)
            if not image_data:
                print(f"🔍 DEBUG: Image {img_index + 1}: Skipped - extraction failed")
                continue
            
            image = image_data['image']
            extraction_method = image_data['method']
            
            # STEP 3: Apply color correction BEFORE validation
            # This fixes inverted images before we check if they're decorative
            corrected_image = _apply_color_corrections(image, img_index + 1)
            
            # STEP 4: Enhanced validation pipeline (now on corrected image)
            validation_result = _validate_image_comprehensively(corrected_image, img_index + 1)
            if not validation_result['is_valid']:
                print(f"🔍 DEBUG: Image {img_index + 1}: Skipped - {validation_result['reason']}")
                continue
            
            # STEP 5: Convert to base64 for LLM processing
            img_base64 = _image_to_base64(corrected_image)
            
            # STEP 6: Get description from vision LLM with enhanced prompts
            print(f"🤖 Image {img_index + 1}: Calling vision LLM...")
            description = describe_educational_image_with_base64(img_base64)
            
            # STEP 7: Final LLM-based decorative check
            if description and not _is_llm_identified_decorative(description):
                relevant_image_count += 1
                print(f"✅ Image {img_index + 1} is RELEVANT - proceeding with save and description")
                
                # Save the image if output directory is provided (only for relevant images)
                saved_image_path = None
                if save_images:
                    # Convert corrected image back to bytes for saving
                    img_bytes = BytesIO()
                    corrected_image.save(img_bytes, format='PNG')
                    saved_image_path = save_image_to_folder(
                        img_bytes.getvalue(), 
                        pdf_name, 
                        relevant_image_count - 1, 
                        output_dir
                    )
                
                # Format description based on number of relevant images
                if relevant_image_count > 1:
                    formatted_description = f"Image {relevant_image_count}: {description}"
                else:
                    formatted_description = f"Image description: {description}"
                
                image_descriptions.append(formatted_description)
                print(f"✅ Image {img_index + 1}: LLM response received ({len(description)} chars)")
                
            else:
                if description:
                    print(f"🔍 DEBUG: Image {img_index + 1} is LLM-identified as DECORATIVE - skipping")
                else:
                    print(f"⚠️ Image {img_index + 1}: Empty LLM response - skipping")
                continue
                
        except Exception as img_error:
            print(f"❌ Image {img_index + 1}: Processing failed - {str(img_error)}")
            continue
    
    print(f"Found {relevant_image_count} relevant images out of {total_images} total images")
    return image_descriptions

# ======================
# PDF Reader function (unchanged - your existing pipeline)
# ======================

def extract_from_pdf(pdf_path: str, output_dir: str = None) -> str:
    """Extracts text and image descriptions from a single PDF."""
    text_extract = ""
    print("Inside extract_from_pdf")
    
    # Get PDF name for image naming
    pdf_name = os.path.basename(pdf_path)
    
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                # Extract text from page
                print(f"Extracting text from page {page_num + 1}")
                text = page.get_text()
                
                # Extract image descriptions (now with ENHANCED filtering and processing)
                # Pass the output_dir so images get saved in the main folder's 'images' subfolder
                image_descriptions = extract_image_descriptions_from_page(
                    page, 
                    pdf_name=pdf_name, 
                    output_dir=output_dir
                )
                # made the image to go below the text
                page_text = text + "\n"
                
                # Prepend image descriptions (if any) to text
                if image_descriptions:
                    combined_description = "\n".join(image_descriptions)
                    page_text += f"\n[Image Descriptions]:\n{combined_description}\n"
                
                # Append to overall question text
                if page_num > 0:
                    text_extract += "\n\n" + page_text
                else:
                    text_extract = page_text
                    
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        text_extract = f"[Error reading PDF: {str(e)}]"
    
    return text_extract.strip()

if __name__ == "__main__":
    description = extract_from_pdf("/Users/harshraj/Downloads/Assignment-1-Finite Element Methods.pdf")
    print(description)