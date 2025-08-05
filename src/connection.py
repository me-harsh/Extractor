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
# Image Descriptor function
# ======================

client = OpenAI(
    # api_key=os.getenv("OPENAI_API_KEY"),
    # base_url=os.getenv("IMAGE_VLLM_URL") 
    api_key="ahosi", #remove this line
    base_url="http://localhost:8011/v1" #remove this line
)

def describe_educational_image_with_base64(base64_image):
    """Describe an image given its base64 encoding using the correct LLM structure."""
    print("called llm for description")#remove this
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
# Image Extractor function
# ======================

def extract_image_descriptions_from_page(page):
    """Extracts images from the page and gets descriptions using the correct LLM calling structure."""
    print("inside extract_image_description_from_page")
    images = list(page.get_images(full=True))
    total_images = len(images)
    image_descriptions = []
    
    for img_index, img in enumerate(images):
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Convert to PIL Image
            image = Image.open(BytesIO(image_bytes))
            
            # Encode image to base64
            buffered = BytesIO()
            # Convert to PNG format for consistency (since the LLM expects PNG in the data URL)
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Call LLM using the correct structure
            description = describe_educational_image_with_base64(img_base64)
            
            # Format description based on number of images
            if total_images > 1:
                formatted_description = f"Image {img_index + 1}: {description}"
            else:
                formatted_description = f"Image description: {description}"
            
            image_descriptions.append(formatted_description)
            
        except Exception as e:
            print(f"Error processing image {img_index}: {str(e)}")
            if total_images > 1:
                image_descriptions.append(f"Image {img_index + 1}: [Image processing failed]")
            else:
                image_descriptions.append("Image description: [Image processing failed]")
    
    return image_descriptions

# ======================
# PDF Reader function
# ======================

def extract_from_pdf(pdf_path: str) -> str:
    """Extracts text and image descriptions from a single PDF."""
    text_extract = ""
    print("inside extract_from_pdf")#remove this
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                # Extract text from page
                print("extracting text")
                text = page.get_text()
                
                # Extract image descriptions
                image_descriptions = extract_image_descriptions_from_page(page)
                
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