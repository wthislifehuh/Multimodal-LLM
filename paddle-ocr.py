from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Run OCR on image
results = ocr.ocr("./assets/waybill.png", cls=True)

# Extract text & bounding box coordinates
for result in results[0]:
    bbox, text, prob = result[0], result[1][0], result[1][1]
    x, y, w, h = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0] - bbox[0][0]), int(bbox[2][1] - bbox[0][1])
    print(f"Text: {text}, ROI: ({x}, {y}, {w}, {h})")