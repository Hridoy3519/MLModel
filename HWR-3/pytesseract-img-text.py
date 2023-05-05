import pytesseract
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage import io
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_text_from_image(img_url):

    img = io.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    text = pytesseract.image_to_string(img)
    return text

