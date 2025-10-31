# გაითვალისწინე რო სათაურს ვერ არჩევს უშუალოდ ტექსტისგან

from PIL import Image
import pytesseract
# თვითონ ფოტოდან ტექსტის ამოგება
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


# C:\ მერე უნდა დაწერო სადაა ფოტო რომელიდანაც გინდა ტექსტის ამოღება C:\Program Files\... ან C:\Users\user\OneDrive\Desktop\ფოტო.jpg
image_path = r"Pics/no_borders.jpg"
result_text = extract_text_from_image(image_path)
# 15/16 აერთებს სხვადასხვა ხაზზე დაწერილ ტექსტს ერთ ხაზად
merged = result_text.split('\n')
result = ' '.join(merged)

# print("Extracted Text:")
# print(result)
