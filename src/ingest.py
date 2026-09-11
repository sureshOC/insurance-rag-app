import pymupdf  # Import the library (historically imported as 'fitz')
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.clean_text import clean_text, forward_fill_table, sanitize_unicode, table_to_text, extract_clause_refs
import uuid


splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def get_all_files(dir='./documents'):
    """
    Get all the files from the document directory.
    input(str): dirctory name
    output(list): list of documented files
    """
    directory_path = Path(dir)
    # Filter only files (ignores folders)
    files = [f for f in directory_path.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']
    print(files)
    return files

def read_file(file_name):
    all_chunks = []
    with pymupdf.open(file_name) as doc:
        for page_num, page in enumerate(doc, start=1):
            print(f"Processing page {page_num}/{len(doc)}...")

            # --- Table Extraction ---
            table_finder = page.find_tables()
            
            # Iterate through all detected tables on the page
            for tab in table_finder:
                raw_table = tab.extract()  # Extract rows as lists
                if raw_table:
                    # Clean text inside every table cell
                    cleaned_table = [[sanitize_unicode(str(cell)) if cell is not None else "" for cell in row]
                        for row in raw_table
                    ]
                    filled_table = forward_fill_table(cleaned_table)
                    table_text = table_to_text(filled_table)
                    if table_text.strip():
                        all_chunks.append({
                            'chunk_id': str(uuid.uuid4()),
                            'text': table_text,
                            'file_name': str(file_name),
                            'page': page_num,
                            'chunk_type': 'table',
                            'clause_refs': [],
                            'method': 'text'
                        })

            text = clean_text(page.get_text())
            method = 'text'

            if not text or len(text) < 20:
                # OCR just this page, not the whole doc
                images = convert_from_path(file_name, dpi=300, first_page=page_num, last_page=page_num)
                if images:
                    raw_ocr = pytesseract.image_to_string(images[0])
                    method = 'ocr'
                    text = clean_text(raw_ocr)

            if text and text.strip():
                split_texts = splitter.split_text(text)
                for snippet in split_texts:
                    all_chunks.append({
                        'chunk_id': str(uuid.uuid4()),
                        'text': snippet,
                        'file_name': str(file_name),
                        'page': page_num,
                        'chunk_type': 'prose',
                        'clause_refs': extract_clause_refs(snippet),
                        'method': method
                    })
                
    return all_chunks

def read_all_data(list_of_files=None):
    data = []
    for file in list_of_files:
        try:
            doc_pages = read_file(file)
            data.extend(doc_pages)
        except Exception as e:
            print(f"FAILED: {file} — {e}")
    return data


# if __name__ == "__main__":
#     all_files = get_all_files()
#     data_chunks = read_all_data(all_files)

#     # import re
#     # for f in set(d['file_name'] for d in data):
#     #     sample = ' '.join(d['data'] for d in data if d['file_name']==f and d['method'] != 'table')
#     #     weird = re.findall(r'[^\x00-\x7F]', sample)
#     #     print(f, set(weird))

#     # print(f"Total pages extracted: {len(data)}")
#     # from collections import Counter
#     # print(Counter(d['file_name'] for d in data))
#     # print(Counter(d['method'] for d in data))
