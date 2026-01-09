from pypdf import PdfReader
import docx
import re

def extract_text_from_pdf(file_path: str) -> str:
    try:
        full_text = ""
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)

            for page in reader.pages:
                full_text += page.extract_text() or ""

        return full_text

    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to extract PDF text: {e}")




def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)

        return '\n'.join(full_text)

    except FileNotFoundError:
        raise FileNotFoundError(f"DOCX file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX text: {e}")

def clean_text(text: str) -> str:

    cleaned_text = re.sub(r'\s+', ' ', text)

    cleaned_text = cleaned_text.strip()

    return cleaned_text


def process_document(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)

    elif file_path.endswith(".docx"):
        raw_text = extract_text_from_docx(file_path)

    else:
        raise ValueError("Only Word doc (.docx) or pdf (.pdf) is accepted!")

    return clean_text(raw_text)


