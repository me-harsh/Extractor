import os
from datetime import datetime
from src.pdf_processing import load_pdf_with_fitz, extract_text_with_fitz
from src.pattern_extractor import EnhancedPatternExtractor
from src.llm_client import LocalLLMClient

def verify_extracted_pdf_content(pdf_path, llm_client):
    """Verify if PDF contains valid question(s) and count them"""
    try:
        # Load the PDF
        doc = load_pdf_with_fitz(pdf_path)
        if not doc:
            return {"status": "error", "message": "Could not load PDF"}
        
        # Extract all text from PDF
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text, _ = extract_text_with_fitz(page)
            full_text += text + "\n"
        
        doc.close()
        
        if not full_text.strip():
            return {"status": "dirt", "message": "No text found in PDF"}
        
        # LLM prompt to verify content with SET detection
        prompt = f"""Analyze this extracted PDF content and determine:
1. Does it contain actual question(s)? (not just headers, page numbers, or junk)
2. How many distinct MAIN questions are present?

CONTENT TO ANALYZE:
{full_text}

Respond in this exact format:
STATUS: [valid/dirt]
QUESTION_COUNT: [number]
REASON: [brief explanation]

Guidelines:
- IMPORTANT: Make sure that even if there is a minor change(such as a simple change in number) in the question, we consider it different questions.
- STATUS should be "valid" if there are actual questions, "dirt" if it's just headers/noise
- QUESTION_COUNT should be the TOTAL number of question variants including SETs
- If STATUS is "dirt", set QUESTION_COUNT to 0
- Consider sub-questions (a), (b), (i), (ii) as part of their main question
- IMPORTANT: Count SETs as separate questions - "Q1 SET A" and "Q1 SET B" are 2 different questions
- Look for patterns like "SET A", "SET B", "Set-A", "Set-B", or similar groupings
- If you see multiple SETs, multiply: 3 questions × 2 sets = 6 total question variants

Examples:
- "Q1 SET A, Q2 SET A, Q1 SET B, Q2 SET B" = QUESTION_COUNT: 4 (2 questions × 2 sets)
- "Q1, Q2, Q3" with no sets = QUESTION_COUNT: 3
- Just headers/page numbers = QUESTION_COUNT: 0"""

        response = llm_client.call_llm(prompt, max_tokens=200)
        
        # Parse LLM response
        status = "dirt"
        question_count = 0
        reason = "Unknown"
        
        if response:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('STATUS:'):
                    status = line.replace('STATUS:', '').strip().lower()
                elif line.startswith('QUESTION_COUNT:'):
                    try:
                        question_count = int(line.replace('QUESTION_COUNT:', '').strip())
                    except ValueError:
                        question_count = 0
                elif line.startswith('REASON:'):
                    reason = line.replace('REASON:', '').strip()
        
        return {
            "status": status,
            "question_count": question_count,
            "reason": reason,
            "text_length": len(full_text.strip())
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Verification failed: {e}"}


def split_multi_question_pdf(pdf_path, llm_client, output_dir):
    """Split PDF containing multiple questions into separate PDFs"""
    try:
        print(f"Splitting multi-question PDF: {os.path.basename(pdf_path)}")
        
        # Load PDF and use existing pattern extraction
        doc = load_pdf_with_fitz(pdf_path)
        if not doc:
            return []
        
        # Reuse existing pattern extractor
        extractor = EnhancedPatternExtractor(llm_client)
        question_starters_info = extractor.extract_main_question_starters_per_page(doc)
        
        if len(question_starters_info) <= 1:
            doc.close()
            return []  # No splitting needed
        
        # Find question positions using existing robust matching
        all_matches = extractor.find_question_starter_positions_with_multiple_strategies(doc, question_starters_info)
        
        if len(all_matches) <= 1:
            doc.close()
            return []  # No splitting needed
        
        print(f"Found {len(all_matches)} questions to split")
        
        # Create output directory for splits
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        split_dir = os.path.join(output_dir, f"{base_name}_split")
        os.makedirs(split_dir, exist_ok=True)
        
        # Use existing function to save split PDFs
        pdf_files = save_multipage_questions_as_pdfs(doc, all_matches, split_dir)
        doc.close()
        
        return pdf_files
        
    except Exception as e:
        print(f"Error splitting PDF {pdf_path}: {e}")
        return []


def verify_and_process_output_directory(output_dir, model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", 
                                      base_url="http://localhost:9091/v1"):
    """Main function to verify all PDFs in output directory and split multi-question ones"""
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return
    
    # Initialize LLM client
    llm_client = LocalLLMClient(base_url=base_url, model=model)
    
    # Get all PDF files in output directory
    pdf_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith('.pdf') and not file.startswith('.'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print(f"No PDF files found in {output_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to verify in {output_dir}")
    
    # Create verification report
    report_file = os.path.join(output_dir, f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    valid_files = []
    dirt_files = []
    split_files = []
    error_files = []
    
    with open(report_file, 'w', encoding='utf-8') as report:
        report.write("PDF VERIFICATION REPORT\n")
        report.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"Directory: {output_dir}\n")
        report.write("=" * 70 + "\n\n")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Verifying: {os.path.basename(pdf_path)}")
            
            # Verify content
            result = verify_extracted_pdf_content(pdf_path, llm_client)
            
            report.write(f"File: {os.path.basename(pdf_path)}\n")
            report.write(f"Status: {result['status']}\n")
            report.write(f"Question Count: {result.get('question_count', 0)}\n")
            report.write(f"Reason: {result.get('reason', 'N/A')}\n")
            report.write(f"Text Length: {result.get('text_length', 0)} chars\n")
            
            if result['status'] == 'valid':
                question_count = result.get('question_count', 1)
                
                if question_count == 1:
                    valid_files.append(pdf_path)
                    print(f"  ✓ Valid single question")
                    report.write("Action: Keep as is\n")
                    
                elif question_count > 1:
                    print(f"  ⚠ Contains {question_count} questions - splitting...")
                    report.write(f"Action: Splitting into {question_count} separate files\n")
                    
                    # Split the PDF
                    split_pdfs = split_multi_question_pdf(pdf_path, llm_client, output_dir)
                    
                    if split_pdfs:
                        split_files.extend(split_pdfs)
                        print(f"  ✓ Split into {len(split_pdfs)} files")
                        report.write(f"Split Result: Created {len(split_pdfs)} files\n")
                        
                        # Optionally move or mark original as processed
                        processed_dir = os.path.join(output_dir, "_processed_originals")
                        os.makedirs(processed_dir, exist_ok=True)
                        processed_path = os.path.join(processed_dir, os.path.basename(pdf_path))
                        
                        # Move original to processed folder
                        try:
                            os.rename(pdf_path, processed_path)
                            report.write(f"Original moved to: {processed_path}\n")
                        except Exception as e:
                            report.write(f"Could not move original: {e}\n")
                    else:
                        print(f"  ✗ Splitting failed")
                        valid_files.append(pdf_path)  # Keep as valid if splitting fails
                        report.write("Split Result: Failed, keeping original\n")
                
            elif result['status'] == 'dirt':
                dirt_files.append(pdf_path)
                print(f"  ✗ Contains no valid questions")
                report.write("Action: Mark as dirt\n")
                
            else:  # error
                error_files.append(pdf_path)
                print(f"  ✗ Verification error: {result.get('message', 'Unknown error')}")
                report.write(f"Action: Error - {result.get('message', 'Unknown')}\n")
            
            report.write("-" * 50 + "\n")
        
        # Write summary
        report.write(f"\nSUMMARY:\n")
        report.write(f"Total files processed: {len(pdf_files)}\n")
        report.write(f"Valid single questions: {len(valid_files)}\n")
        report.write(f"Files split: {len(split_files) // max(1, len([f for f in pdf_files if 'split' not in f]))}\n")
        report.write(f"New files created by splitting: {len(split_files)}\n")
        report.write(f"Dirt files: {len(dirt_files)}\n")
        report.write(f"Error files: {len(error_files)}\n")
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {len(pdf_files)}")
    print(f"Valid single questions: {len(valid_files)}")
    print(f"New files created by splitting: {len(split_files)}")
    print(f"Dirt files: {len(dirt_files)}")
    print(f"Error files: {len(error_files)}")
    print(f"Verification report: {report_file}")
    
    # Optionally move dirt files to separate folder
    if dirt_files:
        dirt_dir = os.path.join(output_dir, "_dirt_files")
        os.makedirs(dirt_dir, exist_ok=True)
        print(f"\nMoving {len(dirt_files)} dirt files to {dirt_dir}/")
        for dirt_file in dirt_files:
            try:
                dirt_dest = os.path.join(dirt_dir, os.path.basename(dirt_file))
                os.rename(dirt_file, dirt_dest)
            except Exception as e:
                print(f"Could not move {dirt_file}: {e}")
    
    return {
        'valid_files': valid_files,
        'split_files': split_files,
        'dirt_files': dirt_files,
        'error_files': error_files,
        'report_file': report_file
    }

