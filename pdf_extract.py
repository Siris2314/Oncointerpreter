import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from io import BytesIO

def convert_pdf_to_text(pdf_path, dpi=500):
    """
    Converts a PDF file to a list of text strings, one per page.

    :param pdf_path: Path to the PDF file.
    :param dpi: Resolution for the conversion.
    :return: List of text strings extracted from each page.
    """
    pages = convert_from_path(pdf_path, dpi)
    extracted_texts = []

    for page_number, page_data in enumerate(pages):
        buffer = BytesIO()
        page_data.save(buffer, format="JPEG")
        buffer.seek(0)

        txt = pytesseract.image_to_string(Image.open(buffer))
        extracted_texts.append(txt)

    return extracted_texts

