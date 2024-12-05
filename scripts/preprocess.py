import os
import pandas as pd
import pytesseract
from PIL import Image
import whisper
from PyPDF2 import PdfReader

def preprocess_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def preprocess_pdf(file_path):
    reader = PdfReader(file_path)
    return ' '.join([page.extract_text() for page in reader.pages])

def preprocess_image(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

def preprocess_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

def preprocess_data(data_dir):
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            ext = file.split('.')[-1]
            file_path = os.path.join(root, file)
            try:
                if ext in ['txt']:
                    content = preprocess_text(file_path)
                elif ext in ['pdf']:
                    content = preprocess_pdf(file_path)
                elif ext in ['png', 'jpg', 'jpeg']:
                    content = preprocess_image(file_path)
                elif ext in ['mp3', 'wav', 'mp4']:
                    content = preprocess_audio(file_path)
                else:
                    continue
                data.append({'file_path': file_path, 'content': content})
            except Exception as e:
                print(f"Failed to process file {file_path}: {e}")
    if len(data) == 0:  # Handle cases where no content was extracted
        return pd.DataFrame(columns=['file_path', 'content'])
    return pd.DataFrame(data)