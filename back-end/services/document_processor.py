import re
from typing import Final

from pypdf import PdfReader
import docx


class DocumentProcessor:

    SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = (".pdf", ".docx")

    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            full_text = ""

            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    full_text += page.extract_text() or ""

            return full_text

        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to extract PDF text: {e}")

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            full_text = []

            for para in doc.paragraphs:
                full_text.append(para.text)

            return "\n".join(full_text)

        except FileNotFoundError:
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to extract DOCX text: {e}")

    @staticmethod
    def clean_text(text: str) -> str:

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def process_document(self, file_path: str) -> str:
        file_path_lower = file_path.lower()

        if file_path_lower.endswith(".pdf"):
            raw_text = self.extract_text_from_pdf(file_path)

        elif file_path_lower.endswith(".docx"):
            raw_text = self.extract_text_from_docx(file_path)

        else:
            raise ValueError(
                f"Unsupported file type. "
                f"Supported types: {self.SUPPORTED_EXTENSIONS}"
            )

        return self.clean_text(raw_text)
