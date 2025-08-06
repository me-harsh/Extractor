from src.pdf_processing import load_pdf_with_fitz
from src.llm_client import LocalLLMClient
from src.pattern_extractor import EnhancedPatternExtractor
from src.question_processor import process_all_pdfs_with_verification

def main():
    """Main entry point for the question extraction pipeline"""
    output = process_all_pdfs_with_verification(model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", 
                                     base_url="http://localhost:9017/v1")
    
    print("$"*50)
    print("the extracted string")
    print("$"*50)
    print(output)

if __name__ == "__main__":
    main()