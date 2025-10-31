import pytesseract
import cv2
from PIL import Image
import preprocess
import extract
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialization of simplifier
INSTRUCTION = "summarize, simplify, and contextualize: "
tokenizer = AutoTokenizer.from_pretrained(
    "haining/scientific_abstract_simplification")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "haining/scientific_abstract_simplification")
input_text = ""


im_path = 'Pics/test1.jpg'
img = cv2.imread(im_path)

# Binarize
gray_image = preprocess.grayscale(img)
cv2.imwrite('Pics/gray.jpg', gray_image)
im_bw = cv2.adaptiveThreshold(
    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5)
cv2.imwrite("Pics/bw_image.jpg", im_bw)



# Border Removal
no_borders = preprocess.remove_borders(im_bw)
cv2.imwrite("Pics/no_borders.jpg", no_borders)

# De-Noise
no_noise = preprocess.noise_removal(no_borders)
cv2.imwrite("Pics/no_noise.jpg", no_noise)

fixed = preprocess.deskew(no_noise)
cv2.imwrite("Pics/rotated_fixed.jpg", fixed)

result_text = extract.extract_text_from_image("Pics/rotated_fixed.jpg")
merged = result_text.split('\n')
result = ' '.join(merged)
#print("Extracted:")
#print(result)

input_text = result
encoding = tokenizer(INSTRUCTION + input_text,
                     max_length=672,
                     padding='max_length',
                     truncation=True,
                     return_tensors='pt')
decoded_ids = model.generate(input_ids=encoding['input_ids'],
                             attention_mask=encoding['attention_mask'],
                             max_length=512,
                             top_p=.9,
                             do_sample=True)
summary = tokenizer.decode(decoded_ids[0], skip_special_tokens=True)

print(summary)

