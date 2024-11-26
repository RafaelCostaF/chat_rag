import fitz  # PyMuPDF

def load_documents(paths):
    """
    Extract text from a list of PDF paths.

    Parameters:
        paths (list of str): List of paths to PDF files.

    Returns:
        list of str: A list of extracted text, one item per PDF.
    """
    extracted_texts = []

    for path in paths:
        try:
            # Open the PDF file
            with fitz.open(path) as pdf:
                text = ""
                # Extract text from each page
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    text += page.get_text() + "\n"
                extracted_texts.append(text.strip())
        except Exception as e:
            print(f"Error reading {path}: {e}")
            extracted_texts.append(None)

    return extracted_texts
