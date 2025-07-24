import re
from src.pdf_processing import extract_text_with_fitz

class EnhancedPatternExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def extract_main_question_starters_per_page(self, doc):
        """Extract main question starters page by page to catch all questions"""
        all_question_starters = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text, _ = extract_text_with_fitz(page)
            
            if not text.strip():
                continue
                
            print(f"\nAnalyzing page {page_num + 1} for question starters...")
            
            prompt = f"""Analyze this text from page {page_num + 1} of a question paper and find ONLY the MAIN questions, not sub-questions or parts.

ACTUAL TEXT FROM PAGE {page_num + 1}:
{text}

Your task:
- Identify ONLY the MAIN questions (typically numbered Q.1, Q.2, etc. or marked with SECTION)
- Do NOT include sub-questions like (a), (b), (i), (ii), Part (A), Part (B), etc.
- Do NOT include section headers, general instructions, or examples
- If a question has multiple sets (Set A, Set B), treat them as separate MAIN questions
- Include both numbered and unnumbered main questions if they are clearly main questions
- Preserve the exact original wording of the question starts

Format your output strictly like this:
QUESTION_START: [first 4-6 words of the main question]

Example outputs:
QUESTION_START: Q.1 Explain the concept
QUESTION_START: Q.2 (a) Define the term
QUESTION_START: SECTION A: Answer
QUESTION_START: 1. What are the
QUESTION_START: Question 3: Discuss the
QUESTION_START: IV. Analyze the following
QUESTION_START: (b) Differentiate between
QUESTION_START: Q.5 Set A:
QUESTION_START: *5. Describe with
QUESTION_START: (10) Explain why

Important notes:
- Include the question number/identifier if present
- Capture exactly how it appears in the text
- If a question continues on next line after number, include those first words
- Don't include questions that are clearly examples or practice questions
- For Roman numeral questions, include the numeral
- For questions in parentheses, include the parentheses

Be precise and extract only actual main questions from this page. Whatever is not present in this page, you do not need to mention, we are doing it page by page, so we will spot it. Your sole job is to identify only on this page."""


            response = self.llm.call_llm(prompt, max_tokens=400)
            print(f"Page {page_num + 1} LLM Response:")
            print(response)
            print("-" * 30)
            
            # Parse question starters from LLM response
            page_starters = []
            if response:
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('QUESTION_START:'):
                        starter = line.replace('QUESTION_START:', '').strip()
                        if starter and len(starter) > 3:
                            page_starters.append({
                                'starter': starter,
                                'page': page_num,
                                'words': starter.split()[:6]  # Keep first 6 words for matching
                            })
            
            print(f"Found {len(page_starters)} question starters on page {page_num + 1}")
            for starter_info in page_starters:
                print(f"  - '{starter_info['starter']}'")
            
            all_question_starters.extend(page_starters)
        
        print(f"\nTotal question starters found across all pages: {len(all_question_starters)}")
        return all_question_starters
    
    def create_flexible_regex_patterns(self, starter_words):
        """Create multiple flexible regex patterns for robust matching"""
        patterns = []
        
        # Pattern 1: Exact sequence with flexible whitespace
        escaped_words = [re.escape(word) for word in starter_words]
        pattern1 = r'\s*'.join(escaped_words)
        patterns.append(('exact_flexible', pattern1))
        
        # Pattern 2: Allow for minor variations (dots, spaces, etc.)
        flexible_words = []
        for word in starter_words:
            # Make dots optional, allow flexible spacing
            flexible_word = re.escape(word).replace(r'\.', r'\.?')
            flexible_words.append(flexible_word)
        pattern2 = r'\s*'.join(flexible_words)
        patterns.append(('flexible_dots', pattern2))
        
        # Pattern 3: First 3 words only (in case LLM extracted too much)
        if len(starter_words) >= 3:
            short_words = [re.escape(word) for word in starter_words[:3]]
            pattern3 = r'\s*'.join(short_words)
            patterns.append(('first_3_words', pattern3))
        
        # Pattern 4: Question number + first significant word
        if len(starter_words) >= 2:
            q_pattern = []
            first_word = starter_words[0]  # Usually Q1, Q2, etc.
            
            # Handle Q1, Q.1, Q1., etc.
            if 'Q' in first_word.upper():
                q_num = re.sub(r'[^\d]', '', first_word)  # Extract number
                if q_num:
                    # Match Q1, Q.1, Q1., Q 1, etc.
                    q_pattern.append(f"Q\\.?\\s*{q_num}\\.?")
                    if len(starter_words) > 1:
                        second_word = re.escape(starter_words[1])
                        pattern4 = f"Q\\.?\\s*{q_num}\\.?\\s+{second_word}"
                        patterns.append(('q_number_flexible', pattern4))
        
        return patterns
    
    def find_question_starter_positions_with_multiple_strategies(self, doc, question_starters_info):
        """Find question starters using multiple matching strategies for maximum robustness"""
        all_matches = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"\nScanning page {page_num + 1} for question starters...")
            
            # Get text with position information
            text_dict = page.get_text("dict")
            
            # Extract all text blocks with their positions
            text_blocks = []
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        for span in line["spans"]:
                            line_text += span["text"]
                            if line_bbox is None:
                                line_bbox = span["bbox"]
                            else:
                                # Expand bbox to include all spans
                                line_bbox = [
                                    min(line_bbox[0], span["bbox"][0]),
                                    min(line_bbox[1], span["bbox"][1]),
                                    max(line_bbox[2], span["bbox"][2]),
                                    max(line_bbox[3], span["bbox"][3])
                                ]
                        
                        if line_text.strip():
                            text_blocks.append({
                                'text': line_text.strip(),
                                'bbox': line_bbox,
                                'y_pos': line_bbox[1] if line_bbox else 0,
                                'y_bottom': line_bbox[3] if line_bbox else 0,
                                'page': page_num,
                                'page_height': page.rect.height
                            })
            
            # Sort by Y position
            text_blocks.sort(key=lambda x: x['y_pos'])
            
            # Find question starter matches on this page using multiple strategies
            for starter_info in question_starters_info:
                starter_text = starter_info['starter']
                starter_words = starter_info['words']
                
                print(f"\n  Searching for: '{starter_text}' on page {page_num + 1}")
                
                # Create multiple regex patterns
                patterns = self.create_flexible_regex_patterns(starter_words)
                
                found_match = False
                
                # Try each pattern until we find a match
                for pattern_name, pattern in patterns:
                    if found_match:
                        break
                        
                    print(f"    Trying pattern '{pattern_name}': {pattern}")
                    
                    for block in text_blocks:
                        block_text = block['text'].strip()
                        
                        # Use regex search (more flexible than findall)
                        match = re.search(pattern, block_text, re.IGNORECASE)
                        
                        if match:
                            match_info = {
                                'starter': starter_text,
                                'matched_text': block_text,
                                'page': page_num,
                                'y_pos': block['y_pos'],
                                'y_bottom': block['y_bottom'],
                                'bbox': block['bbox'],
                                'page_height': page.rect.height,
                                'absolute_position': page_num * 1000 + block['y_pos'],
                                'pattern_used': pattern_name,
                                'regex_pattern': pattern
                            }
                            all_matches.append(match_info)
                            print(f"    ✓ FOUND with pattern '{pattern_name}' at Y={block['y_pos']:.2f}")
                            print(f"      Matched text: '{block_text[:60]}...'")
                            found_match = True
                            break
                
                if not found_match:
                    print(f"    ✗ No match found for '{starter_text}' on page {page_num + 1}")
                    
                    # FALLBACK: Try simple substring matching
                    print(f"    Trying fallback substring matching...")
                    first_word = starter_words[0] if starter_words else ""
                    
                    if first_word:
                        for block in text_blocks:
                            block_text = block['text'].strip()
                            
                            # Try simple substring matching
                            if first_word.lower() in block_text.lower():
                                match_info = {
                                    'starter': starter_text,
                                    'matched_text': block_text,
                                    'page': page_num,
                                    'y_pos': block['y_pos'],
                                    'y_bottom': block['y_bottom'],
                                    'bbox': block['bbox'],
                                    'page_height': page.rect.height,
                                    'absolute_position': page_num * 1000 + block['y_pos'],
                                    'pattern_used': 'substring_fallback',
                                    'regex_pattern': f"substring: {first_word}"
                                }
                                all_matches.append(match_info)
                                print(f"    ✓ FOUND with substring fallback at Y={block['y_pos']:.2f}")
                                print(f"      Matched text: '{block_text[:60]}...'")
                                found_match = True
                                break
                
                if not found_match:
                    print(f"    ✗ Complete failure to find '{starter_text}' on page {page_num + 1}")
        
        # Sort all matches by absolute position (page order + Y position)
        all_matches.sort(key=lambda x: x['absolute_position'])
        
        print(f"\nFound {len(all_matches)} total question matches across all pages")
        
        # Remove duplicates that might be very close to each other
        filtered_matches = []
        for match in all_matches:
            # Check if this match is too close to a previous one
            is_duplicate = False
            for existing in filtered_matches:
                if (existing['page'] == match['page'] and 
                    abs(existing['y_pos'] - match['y_pos']) < 10):
                    is_duplicate = True
                    print(f"  Removing duplicate at page {match['page']+1}, Y={match['y_pos']:.2f}")
                    break
            
            if not is_duplicate:
                filtered_matches.append(match)
        
        print(f"After removing duplicates: {len(filtered_matches)} unique questions")
        return filtered_matches