import fitz  # PyMuPDF
from openai import OpenAI
import base64
import requests
from io import BytesIO
from PIL import Image
from typing import List, Dict, Union
import os
from dotenv import load_dotenv
from pathlib import Path

# === CONFIG ===
# IMAGE_VLLM_URL = "http://localhost:9019/v1"
# IMAGE_VLLM_MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"


# ======================
# Image Relevance Check function
# ======================

client = OpenAI(
    # api_key=os.getenv("OPENAI_API_KEY"),
    # base_url=os.getenv("IMAGE_VLLM_URL") 
    api_key="ahosi", #remove this line
    base_url="http://localhost:8011/v1" #remove this line
)

def check_image_relevance_with_base64(base64_image):
    """Check if an image is relevant for educational/academic content."""
    print("Checking image relevance...")
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",#remove this line
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that determines if images are relevant for educational or academic content. "
                    "An image is considered RELEVANT if it contains: diagrams, charts, graphs, mathematical equations, "
                    "scientific illustrations, technical drawings, maps, educational figures, tables with meaningful data, "
                    "flowcharts, circuit diagrams, or any content that supports learning or understanding of academic material.\n\n"
                    "An image is considered NOT RELEVANT if it contains:\n"
                    "- Logos, watermarks, or branding elements\n"
                    "- Decorative elements, borders, or design flourishes\n"
                    "- Headers, footers, or page formatting elements\n"
                    "- Page numbers or page counting information\n"
                    "- Question counting information (e.g., 'No. of Questions = 6')\n"
                    "- Page metadata (e.g., 'No. of Pages = 21')\n"
                    "- Simple text boxes with administrative info like page/question counts\n"
                    "- Blank spaces, white backgrounds, or empty content\n"
                    "- University logos, institutional branding\n"
                    "- Random photos or non-educational images\n"
                    "- Any content that is purely administrative or formatting-related\n\n"
                    "Examples of NOT RELEVANT content:\n"
                    "- 'No. of Pages = 2, No. of Questions = 6'\n"
                    "- University logos or institutional headers\n"
                    "- Page numbers or footers\n"
                    "- Decorative borders or design elements\n\n"
                    "Respond with ONLY 'RELEVANT' or 'NOT_RELEVANT' - no other text."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Is this image relevant for educational/academic content? Answer only 'RELEVANT' or 'NOT_RELEVANT'."
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
        max_tokens=10  # We only need a short response
    )
    
    result = response.choices[0].message.content.strip().upper()
    is_relevant = result == "RELEVANT"
    print(f"Image relevance check result: {result} -> {is_relevant}")
    return is_relevant

# ======================
# Image Descriptor function
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
                    "You are an assistant that describes educational diagrams and images "
                    "so that an illustrator or educator can recreate them precisely. "
                    "Be detailed: include layout, labels, visual elements, relationships, arrows, "
                    "colors, text, and any annotations. Keep the tone clear and structured."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please describe this educational image in detail so that someone could recreate it from scratch. "
                            "Include object positions, shapes, sizes, label names, arrows, styles, fonts, spacing, and any implied relationships."
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
# Image Saving function
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
# Image Extractor function (Modified with 2-step process)
# ======================

def extract_image_descriptions_from_page(page, pdf_name=None, output_dir=None):
    """Extracts images from the page, checks relevance, saves relevant ones, and gets descriptions using the 2-step process."""
    print("Inside extract_image_description_from_page")
    images = list(page.get_images(full=True))
    total_images = len(images)
    image_descriptions = []
    relevant_image_count = 0
    
    print(f"Total images found: {total_images}")
    
    # Only save images if output_dir is provided
    if output_dir is not None:
        save_images = True
        if pdf_name is None:
            pdf_name = "unknown_pdf"  # fallback only if output_dir is provided
    else:
        save_images = False
    
    for img_index, img in enumerate(images):
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Process for relevance check
            # Convert to PIL Image
            image = Image.open(BytesIO(image_bytes))
            
            # Encode image to base64
            buffered = BytesIO()
            # Convert to PNG format for consistency
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # STEP 1: Check if image is relevant
            print(f"Processing image {img_index + 1}/{total_images}")
            is_relevant = check_image_relevance_with_base64(img_base64)
            
            if is_relevant:
                relevant_image_count += 1
                print(f"Image {img_index + 1} is RELEVANT - proceeding with description")
                
                # Save the image if output directory is provided (only for relevant images)
                saved_image_path = None
                if save_images:
                    saved_image_path = save_image_to_folder(image_bytes, pdf_name, relevant_image_count - 1, output_dir)
                
                # STEP 2: Get description for relevant image
                description = describe_educational_image_with_base64(img_base64)
                
                # Format description based on number of relevant images
                if relevant_image_count > 1:
                    formatted_description = f"Image {relevant_image_count}: {description}"
                else:
                    formatted_description = f"Image description: {description}"
                
                image_descriptions.append(formatted_description)
                
            else:
                print(f"Image {img_index + 1} is NOT RELEVANT - skipping")
                # We don't save or describe irrelevant images
                continue
                
        except Exception as e:
            print(f"Error processing image {img_index + 1}: {str(e)}")
            # Only add error message for what would have been a relevant image
            # Since we can't check relevance due to error, we'll skip it
            continue
    
    print(f"Found {relevant_image_count} relevant images out of {total_images} total images")
    return image_descriptions

# ======================
# PDF Reader function (Modified)
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
                
                # Extract image descriptions (now with relevance checking and image saving)
                # Pass the output_dir so images get saved in the main folder's 'images' subfolder
                image_descriptions = extract_image_descriptions_from_page(
                    page, 
                    pdf_name=pdf_name, 
                    output_dir=output_dir
                )
                
                # Prepend image descriptions (if any) to text
                if image_descriptions:
                    combined_description = "\n".join(image_descriptions)
                    page_text = f"[Image Descriptions]:\n{combined_description}\n\n{text}"
                else:
                    page_text = text
                
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