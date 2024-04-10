import os
import re
import cv2
from paddleocr import PaddleOCR

def ocr(frame, ocr_model):
    result = ocr_model.ocr(frame, cls=True)
    text = [re.sub(r'[^0-9.]', '', line[1][0]) for line in result[0] if line[1][0]]
    return text if text else []

def frequent(ocr_results):
    text_frequency = {}
    for texts in ocr_results:
        for text in texts:
            text_frequency[text] = text_frequency.get(text, 0) + 1
    return [text for text, count in text_frequency.items() if count >= 3]

def save_image(image, ocr_texts, save_path):
    for i, text in enumerate(ocr_texts):
        cv2.putText(image, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imwrite(save_path, image)

def process_video(img_path, save_path):
    result_ocr_list = []
    result_frame = []
    cap = cv2.VideoCapture(img_path)
    ocr_model = PaddleOCR(use_gpu=True) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ocr_result = ocr(frame, ocr_model) 
        result_ocr_list.append(ocr_result)
        result_frame.append(frame)
    cap.release()

    result_three = frequent(result_ocr_list)

    for t, ocr_texts in enumerate(result_ocr_list):
        if any(text in result_three for text in ocr_texts):
            full_save_path = os.path.join(save_path, f"frame_{t}.jpg")
            save_image(result_frame[t], ocr_texts, full_save_path)

    print("Processing is over!")

def main():
    img_path = 'E:\\Python 3.9 Pycharm Community\\pythonProject2\\ocr\\trimmed\\14g_20psi.mov'
    save_path = "E:\\Python 3.9 Pycharm Community\\pythonProject2\\ocr\\ocr\\xiebro"
    process_video(img_path, save_path)

if __name__ == "__main__":
    main()