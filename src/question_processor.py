import os
from datetime import datetime
from src.pdf_processing import load_pdf_with_fitz
from src.pattern_extractor import EnhancedPatternExtractor
from src.llm_client import LocalLLMClient
from src.verification import verify_and_process_output_directory
from src.connection import extract_from_pdf
import fitz  # PyMuPDF      


def create_multipage_question_pdf(source_doc, question_segments, output_path, question_num):
    """Create a PDF containing a question that may span multiple pages"""
    try:
        # Create a new PDF document
        new_doc = fitz.open()
        
        for segment in question_segments:
            page_num = segment['page']
            y_start = segment['y_start']
            y_end = segment['y_end']
            
            # Get the source page
            source_page = source_doc[page_num]
            page_rect = source_page.rect
            
            # Define the crop rectangle for this segment
            crop_rect = fitz.Rect(
                0,  # x0 - start from left edge
                y_start,  # y0 - start Y position
                page_rect.width,  # x1 - full width
                min(y_end, page_rect.height)  # y1 - end Y position (don't exceed page)
            )
            
            # Calculate the height of this segment
            segment_height = crop_rect.height
            
            # Create new page for this segment
            new_page = new_doc.new_page(width=crop_rect.width, height=segment_height)
            
            # Copy the cropped area from source to new page
            new_page.show_pdf_page(
                fitz.Rect(0, 0, crop_rect.width, segment_height),  # target rectangle
                source_doc,  # source document
                page_num,  # source page number
                clip=crop_rect  # area to copy
            )
            
            print(f"  Added segment from page {page_num + 1}, Y {y_start:.2f}-{y_end:.2f} (height: {segment_height:.2f})")
        
        # Save the new PDF
        new_doc.save(output_path)
        new_doc.close()
        
        print(f"Created multi-page question PDF: {output_path}")
        print(f"  Question spans {len(question_segments)} page segment(s)")
        return True
        
    except Exception as e:
        print(f"Error creating multi-page question PDF {output_path}: {e}")
        return False

def determine_question_segments(all_matches):
    """Determine how questions span across pages by analyzing question starter positions"""
    if not all_matches:
        return []
    
    questions = []
    
    for i, current_match in enumerate(all_matches):
        question_segments = []
        
        # Determine where this question ends
        if i + 1 < len(all_matches):
            next_match = all_matches[i + 1]
            
            # Check if question spans multiple pages
            if current_match['page'] == next_match['page']:
                # Same page - simple case
                question_segments.append({
                    'page': current_match['page'],
                    'y_start': max(0, current_match['y_pos'] - 5),  # Small margin above
                    'y_end': next_match['y_pos'] - 2  # Stop just before next question
                })
            else:
                # Multi-page question
                print(f"Multi-page question detected: Q{i+1} spans from page {current_match['page']+1} to page {next_match['page']+1}")
                
                # First segment: from current pattern to end of current page
                question_segments.append({
                    'page': current_match['page'],
                    'y_start': max(0, current_match['y_pos'] - 5),
                    'y_end': current_match['page_height']  # Go to end of page
                })
                
                # Middle segments: any complete pages between start and end
                for middle_page in range(current_match['page'] + 1, next_match['page']):
                    question_segments.append({
                        'page': middle_page,
                        'y_start': 0,  # Start from top of page
                        'y_end': all_matches[0]['page_height']  # Full page height
                    })
                    print(f"  Added full page {middle_page + 1} to question {i+1}")
                
                # Last segment: from top of next page to next pattern
                if next_match['page'] > current_match['page']:
                    question_segments.append({
                        'page': next_match['page'],
                        'y_start': 0,  # Start from top of page
                        'y_end': next_match['y_pos'] - 2  # Stop before next question
                    })
        else:
            # Last question - goes to end of document
            question_segments.append({
                'page': current_match['page'],
                'y_start': max(0, current_match['y_pos'] - 5),
                'y_end': current_match['page_height']
            })
        
        questions.append({
            'question_number': i + 1,
            'starter': current_match['starter'],
            'segments': question_segments,
            'total_pages': len(question_segments),
            'pattern_used': current_match.get('pattern_used', 'unknown')
        })
    
    return questions

def save_multipage_questions_as_pdfs(source_doc, all_matches, output_dir="extracted_questions_multipage"):
    """Save each question as a separate PDF file, handling multi-page questions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine question segments
    questions = determine_question_segments(all_matches)
    
    summary_file = os.path.join(output_dir, "questions_summary.txt")
    pdf_files_created = []
    
    with open(summary_file, 'w', encoding='utf-8') as summary:
        summary.write("MULTI-PAGE QUESTION EXTRACTION SUMMARY\n")
        summary.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary.write(f"Source PDF: {os.path.basename(source_doc.name)}\n")
        summary.write("=" * 50 + "\n\n")
        
        for question in questions:
            # Create PDF filename
            pdf_filename = f"question_{question['question_number']:02d}_multipage.pdf"
            pdf_filepath = os.path.join(output_dir, pdf_filename)
            
            # Create the multi-page PDF
            success = create_multipage_question_pdf(
                source_doc,
                question['segments'],
                pdf_filepath,
                question['question_number']
            )
            
            if success:
                pdf_files_created.append(pdf_filepath)
                
                # Add to summary
                summary.write(f"Question {question['question_number']}:\n")
                summary.write(f"  PDF File: {pdf_filename}\n")
                summary.write(f"  Starter: {question['starter']}\n")
                summary.write(f"  Pattern Used: {question['pattern_used']}\n")
                summary.write(f"  Total Pages: {question['total_pages']}\n")
                summary.write(f"  Page Segments:\n")
                
                for j, segment in enumerate(question['segments']):
                    summary.write(f"    Segment {j+1}: Page {segment['page']+1}, Y {segment['y_start']:.2f}-{segment['y_end']:.2f}\n")
                
                summary.write("\n")
    
    print(f"\nCreated {len(pdf_files_created)} multi-page question PDF files in {output_dir}/")
    print(f"Summary saved to {summary_file}")
    return pdf_files_created

def extract_questions_as_multipage_pdfs(pdf_path, model="gemma3:4b", base_url="http://localhost:11434/v1"):
    """Extract questions and save them as separate PDF files, with ultra-robust pattern matching"""
    print(f"Processing PDF: {pdf_path}")
    print(f"Using model: {model}")
    
    # Initialize
    llm_client = LocalLLMClient(base_url=base_url, model=model)
    extractor = EnhancedPatternExtractor(llm_client)
    
    # Test LLM connection
    print("\nTesting LLM connection...")
    test_response = llm_client.call_llm("What is 2+2?")
    print(f"LLM test response: {test_response}")
    
    # Load PDF with fitz
    doc = load_pdf_with_fitz(pdf_path)
    if not doc:
        return []
    
    total_pages = len(doc)
    
    # Extract main question starters using LLM page by page
    print("\nExtracting main question starters page by page using LLM...")
    question_starters_info = extractor.extract_main_question_starters_per_page(doc)
    
    if not question_starters_info:
        print("No main question starters found! Cannot extract questions.")
        doc.close()
        return [], []
    
    # Find ALL question starter positions using multiple robust strategies
    print("\nScanning entire document for question starter positions with multiple strategies...")
    all_matches = extractor.find_question_starter_positions_with_multiple_strategies(doc, question_starters_info)
    
    if not all_matches:
        print("No question starter matches found in the document!")
        doc.close()
        return [], []
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Create unique output directory with timestamp and PDF name
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/extracted_questions_{pdf_name}_{timestamp}"
    
    # Create multi-page question PDFs in the unique directory
    pdf_files = save_multipage_questions_as_pdfs(doc, all_matches, output_dir)
    
    # Close the document
    doc.close()
    
    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY: Created {len(pdf_files)} multi-page question PDF files from {total_pages} pages")
    print(f"Output directory: {output_dir}/")
    print(f"Ultra-robust pattern matching successfully found {len(all_matches)} questions!")
    print(f"{'='*50}")
    
    return all_matches, pdf_files


def process_all_pdfs_with_verification(model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", 
                                       base_url="http://localhost:8012/v1"):
    """Process all PDFs in test_files and automatically verify/split results"""
    print("STEP 1: Processing all PDFs in test_files directory")
    print("=" * 70)
    
    # Step 1: Create the output PDFs
    process_all_pdfs_in_test_files(model, base_url)
    
    print(f"\nSTEP 2: Auto-verifying all recent output directories")
    print("=" * 70)
    
    base_output_dir = "output"
    if not os.path.exists(base_output_dir):
        print(f"No output directory found: {base_output_dir}")
        return ""
    
    # Get today's date string
    today = datetime.now().strftime("%Y%m%d")
    
    # Collect recent output directories
    extraction_dirs = []
    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        if (os.path.isdir(item_path) and 
            item.startswith("extracted_questions_") and 
            today in item):
            extraction_dirs.append(item_path)
    
    if not extraction_dirs:
        print(f"No recent extraction directories found in {base_output_dir}")
        return ""
    
    print(f"Found {len(extraction_dirs)} recent output directories to verify")
    
    # Step 3: Run verification and collect all valid PDFs
    all_results = []
    all_valid_pdfs = []
    
    for i, output_dir in enumerate(extraction_dirs, 1):
        print(f"\n[{i}/{len(extraction_dirs)}] Verifying: {os.path.basename(output_dir)}")
        print("-" * 50)
        
        results = verify_and_process_output_directory(output_dir, model, base_url)
        all_results.append({
            'directory': output_dir,
            'results': results
        })
        
        # Collect valid PDFs from this verification result
        if results and 'valid_files' in results:
            all_valid_pdfs.extend(results['valid_files'])

    # Step 4: Print summary
    print(f"\n{'='*70}")
    print("FINAL VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    total_valid = sum(len(r['results']['valid_files']) for r in all_results if r['results'])
    total_split = sum(len(r['results']['split_files']) for r in all_results if r['results'])
    total_dirt = sum(len(r['results']['dirt_files']) for r in all_results if r['results'])
    
    print(f"Directories processed: {len(all_results)}")
    print(f"Total valid questions: {total_valid}")
    print(f"Total files from splitting: {total_split}")
    print(f"Total dirt files: {total_dirt}")

    # Step 5: Extract text from valid PDFs
    all_extracted_questions = ""
    print(f"\nExtracting text from {len(all_valid_pdfs)} valid PDF files...")
    
    for out_file in all_valid_pdfs:
        print(f"  - {out_file}")
        try:
            extracted_text = extract_from_pdf(out_file)
            all_extracted_questions += extracted_text + "\n\n"
        except Exception as e:
            print(f"Error extracting from {out_file}: {e}")
    
    return all_extracted_questions

def process_all_pdfs_in_test_files(model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", 
                                 base_url="http://localhost:9091/v1"):
    """Process all PDF files in the test_files directory"""
    # Define the test_files directory path
    test_files_dir = os.path.join(os.getcwd(), "test_files")
    
    # Check if directory exists
    if not os.path.exists(test_files_dir):
        print(f"Directory not found: {test_files_dir}")
        print("Please create a 'test_files' directory in the current working directory")
        return
    
    # Get all PDF files in test_files directory
    pdf_files = [f for f in os.listdir(test_files_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {test_files_dir}!")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process in test_files/:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(test_files_dir, pdf_file)
        print(f"\n{'='*80}")
        print(f"PROCESSING: {pdf_path}")
        print(f"{'='*80}")
        
        try:
            # Extract questions
            matches, output_files = extract_questions_as_multipage_pdfs(
                pdf_path,
                model=model,
                base_url=base_url
            )
            
            # Print summary
            print("\nEXTRACTED QUESTIONS SUMMARY:")
            print("=" * 60)
            for i, match in enumerate(matches):
                print(f"\nQUESTION {i+1}:")
                print(f"Starter: {match['starter']}")
                print(f"Page: {match['page'] + 1}")
                print(f"Y Position: {match['y_pos']:.2f}")
                print(f"Pattern Used: {match.get('pattern_used', 'unknown')}")
                print(f"Matched Text: {match['matched_text'][:100]}...")
                print("-" * 40)
                
            print(f"\nCreated output files for {pdf_file}:")
            for out_file in output_files:
                print(f"  - {out_file}")
                
        except Exception as e:
            print(f"\nERROR processing {pdf_file}: {str(e)}")
            continue

