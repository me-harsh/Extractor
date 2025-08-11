import os
import sys
from pathlib import Path

# Method 1: Using docx2pdf (Recommended - Best formatting preservation)
def convert_docx_to_pdf_simple(docx_path, pdf_path=None):
    """
    Convert DOCX to PDF using docx2pdf library (requires Microsoft Word on Windows)
    This method preserves all formatting, images, tables, etc.
    """
    try:
        from docx2pdf import convert
        
        if pdf_path is None:
            pdf_path = docx_path.replace('.docx', '.pdf')
        
        # Convert single file
        convert(docx_path, pdf_path)
        print(f"Successfully converted {docx_path} to {pdf_path}")
        return pdf_path
        
    except ImportError:
        print("docx2pdf not installed. Install with: pip install docx2pdf")
        return None
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

# Method 2: Using python-docx + reportlab (Cross-platform)
def convert_docx_to_pdf_advanced(docx_path, pdf_path=None):
    """
    Convert DOCX to PDF using python-docx and reportlab
    Preserves text formatting, images, and basic structure
    """
    try:
        from docx import Document
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
        from io import BytesIO
        import tempfile
        
        if pdf_path is None:
            pdf_path = docx_path.replace('.docx', '.pdf')
        
        # Load the DOCX document
        doc = Document(docx_path)
        
        # Create PDF document
        pdf_doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        
        # Get sample styles
        styles = getSampleStyleSheet()
        
        # Create custom styles for different formatting
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=10
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )
        
        # Process each paragraph
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                # Determine style based on paragraph properties
                if paragraph.style.name.startswith('Heading 1'):
                    style = title_style
                elif paragraph.style.name.startswith('Heading'):
                    style = heading_style
                else:
                    style = normal_style
                
                # Handle text alignment
                if paragraph.alignment == 1:  # Center
                    style.alignment = TA_CENTER
                elif paragraph.alignment == 2:  # Right
                    style.alignment = TA_RIGHT
                elif paragraph.alignment == 3:  # Justify
                    style.alignment = TA_JUSTIFY
                else:
                    style.alignment = TA_LEFT
                
                # Process runs for formatting
                formatted_text = ""
                for run in paragraph.runs:
                    text = run.text
                    if run.bold:
                        text = f"<b>{text}</b>"
                    if run.italic:
                        text = f"<i>{text}</i>"
                    if run.underline:
                        text = f"<u>{text}</u>"
                    formatted_text += text
                
                # Add paragraph to story
                if formatted_text.strip():
                    story.append(Paragraph(formatted_text, style))
                    story.append(Spacer(1, 6))
        
        # Process images
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    
                    # Save image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img:
                        temp_img.write(image_data)
                        temp_img_path = temp_img.name
                    
                    # Add image to PDF
                    img = RLImage(temp_img_path, width=4*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                    
                    # Clean up temporary file
                    os.unlink(temp_img_path)
                    
                except Exception as img_error:
                    print(f"Error processing image: {img_error}")
        
        # Build PDF
        pdf_doc.build(story)
        print(f"Successfully converted {docx_path} to {pdf_path}")
        return pdf_path
        
    except ImportError as e:
        print(f"Required library not installed: {e}")
        print("Install with: pip install python-docx reportlab")
        return None
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

# Method 3: Using LibreOffice (if available)
def convert_docx_to_pdf_libreoffice(docx_path, output_dir=None):
    """
    Convert DOCX to PDF using LibreOffice command line
    Excellent formatting preservation, cross-platform
    """
    try:
        import subprocess
        
        if output_dir is None:
            output_dir = os.path.dirname(docx_path)
        
        # LibreOffice command
        cmd = [
            'libreoffice',
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', output_dir,
            docx_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            pdf_path = os.path.join(output_dir, 
                                  os.path.basename(docx_path).replace('.docx', '.pdf'))
            print(f"Successfully converted {docx_path} to {pdf_path}")
            return pdf_path
        else:
            print(f"LibreOffice conversion failed: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("LibreOffice not found. Install LibreOffice for this method.")
        return None
    except Exception as e:
        print(f"Error with LibreOffice conversion: {e}")
        return None

# Method 4: Using pandoc (if available)
def convert_docx_to_pdf_pandoc(docx_path, pdf_path=None):
    """
    Convert DOCX to PDF using pandoc
    Good for text-heavy documents
    """
    try:
        import subprocess
        
        if pdf_path is None:
            pdf_path = docx_path.replace('.docx', '.pdf')
        
        cmd = ['pandoc', docx_path, '-o', pdf_path, '--pdf-engine=xelatex']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully converted {docx_path} to {pdf_path}")
            return pdf_path
        else:
            print(f"Pandoc conversion failed: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("Pandoc not found. Install pandoc for this method.")
        return None
    except Exception as e:
        print(f"Error with pandoc conversion: {e}")
        return None

# Batch conversion function
def batch_convert_docx_to_pdf(input_folder, output_folder=None, method='simple'):
    """
    Convert all DOCX files in a folder to PDF
    """
    if output_folder is None:
        output_folder = input_folder
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    conversion_methods = {
        'simple': convert_docx_to_pdf_simple,
        'advanced': convert_docx_to_pdf_advanced,
        'libreoffice': convert_docx_to_pdf_libreoffice,
        'pandoc': convert_docx_to_pdf_pandoc
    }
    
    convert_func = conversion_methods.get(method, convert_docx_to_pdf_simple)
    
    for docx_file in Path(input_folder).glob('*.docx'):
        pdf_path = Path(output_folder) / f"{docx_file.stem}.pdf"
        
        if method == 'libreoffice':
            convert_func(str(docx_file), output_folder)
        else:
            convert_func(str(docx_file), str(pdf_path))

# Modified main execution function for test_files directory
def convert_all_docx_in_test_files():
    """
    Convert all DOCX files in 'test_files' directory to PDF in the same directory
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_files_dir = os.path.join(script_dir, 'test_files')
    
    # Check if test_files directory exists
    if not os.path.exists(test_files_dir):
        print(f"❌ Error: 'test_files' directory not found at: {test_files_dir}")
        print("Please create the 'test_files' folder in the same directory as this script.")
        return
    
    # Find all DOCX files in test_files directory
    docx_files = list(Path(test_files_dir).glob('*.docx'))
    
    if not docx_files:
        print(f"No DOCX files found in: {test_files_dir}")
        return
    
    print(f"Found {len(docx_files)} DOCX file(s) in test_files directory:")
    for docx_file in docx_files:
        print(f"  - {docx_file.name}")
    
    print(f"\nConverting files in: {test_files_dir}")
    print("=" * 60)
    
    successful_conversions = 0
    failed_conversions = []
    
    for docx_file in docx_files:
        docx_path = str(docx_file)
        pdf_path = str(docx_file.with_suffix('.pdf'))
        
        print(f"\nConverting: {docx_file.name}")
        
        # Try methods in order of preference
        success = False
        
        # Method 1: Simple conversion (docx2pdf)
        print("  Trying docx2pdf...")
        result = convert_docx_to_pdf_simple(docx_path, pdf_path)
        if result:
            successful_conversions += 1
            success = True
            print(f"  ✅ Success: {docx_file.name} → {Path(pdf_path).name}")
            continue
        
        # Method 2: LibreOffice conversion
        print("  Trying LibreOffice...")
        result = convert_docx_to_pdf_libreoffice(docx_path, test_files_dir)
        if result:
            successful_conversions += 1
            success = True
            print(f"  ✅ Success: {docx_file.name} → {Path(pdf_path).name}")
            continue
        
        # Method 3: Advanced conversion (python-docx + reportlab)
        print("  Trying python-docx + reportlab...")
        result = convert_docx_to_pdf_advanced(docx_path, pdf_path)
        if result:
            successful_conversions += 1
            success = True
            print(f"  ✅ Success: {docx_file.name} → {Path(pdf_path).name}")
            continue
        
        # Method 4: Pandoc conversion
        print("  Trying Pandoc...")
        result = convert_docx_to_pdf_pandoc(docx_path, pdf_path)
        if result:
            successful_conversions += 1
            success = True
            print(f"  ✅ Success: {docx_file.name} → {Path(pdf_path).name}")
            continue
        
        if not success:
            failed_conversions.append(docx_file.name)
            print(f"  ❌ Failed: {docx_file.name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Source folder: {test_files_dir}")
    print(f"Total files: {len(docx_files)}")
    print(f"Successful: {successful_conversions}")
    print(f"Failed: {len(failed_conversions)}")
    
    if failed_conversions:
        print(f"\nFailed files:")
        for failed_file in failed_conversions:
            print(f"  - {failed_file}")
        print("\nTo resolve failures, install required dependencies:")
        print("  pip install docx2pdf")
        print("  pip install python-docx reportlab pillow")
        print("  Install LibreOffice: https://www.libreoffice.org/")
    
    if successful_conversions > 0:
        print(f"\n✅ {successful_conversions} PDF file(s) created in: {test_files_dir}")

def main():
    """
    Main function - converts all DOCX files in test_files directory
    """
    print("DOCX to PDF Batch Converter")
    print("Converting all DOCX files in 'test_files' folder...")
    print("=" * 60)
    
    convert_all_docx_in_test_files()

if __name__ == "__main__":
    # Installation commands for different methods:
    print("Installation commands:")
    print("pip install docx2pdf  # Method 1 (Windows with MS Word)")
    print("pip install python-docx reportlab  # Method 2 (Cross-platform)")
    print("# Install LibreOffice from https://www.libreoffice.org/  # Method 3")
    print("# Install pandoc from https://pandoc.org/  # Method 4")
    print("\n" + "="*60)
    
    main()