import pytesseract
from PIL import Image


def extract_image(filename):
    # Load the PNG image
    image = Image.open(filename)

    # Extract text from the image
    text = pytesseract.image_to_string(image)

    with open('report.txt', 'w') as file:
        file.write(text)