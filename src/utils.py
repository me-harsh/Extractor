import os
from datetime import datetime

def create_output_directory(base_name):
    """Create unique output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/extracted_questions_{base_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_pdf_files(directory):
    """Get all PDF files in a directory"""
    return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]