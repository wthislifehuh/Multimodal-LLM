import cv2
from paddleocr import PaddleOCR
import time

start_time = time.time()

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Load the image
image_path = "./assets/waybill.png"
image = cv2.imread(image_path)

# Run OCR on image
results = ocr.ocr(image_path, cls=True)

# Define target words (keys)
target_keys = {"WEIGHT", "NO OF PCS", "POST CODE", "POST CODE:", "ACCOUNT", "ACCOUNT:"}
extracted_data = {}

# Store detected text and their bounding boxes
detected_text = []
for result in results[0]:
    bbox, text, prob = result[0], result[1][0], result[1][1]
    x, y, w, h = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0] - bbox[0][0]), int(bbox[2][1] - bbox[0][1])
    
    detected_text.append({"text": text.upper(), "bbox": (x, y, w, h)})

# Find key-value pairs
for entry in detected_text:
    key_text = entry["text"]
    key_bbox = entry["bbox"]

    if key_text in target_keys:
        # Find the closest value (text located to the right or below)
        closest_value = None
        min_distance = float("inf")

        for other_entry in detected_text:
            value_text = other_entry["text"]
            value_bbox = other_entry["bbox"]

            # Skip if it's the same as key
            if value_text == key_text:
                continue

            # Calculate vertical and horizontal distance
            key_x, key_y, key_w, key_h = key_bbox
            value_x, value_y, value_w, value_h = value_bbox

            dx = abs(value_x - (key_x + key_w))  # Distance to the right
            dy = abs(value_y - key_y)  # Vertical alignment

            # Consider only values to the right or below the key
            if (value_x > key_x or dy < key_h * 2) and dx < min_distance:
                closest_value = other_entry
                min_distance = dx

        # Store key-value pair
        if closest_value:
            extracted_data[key_text] = {
                "value": closest_value["text"],
                "key_roi": key_bbox,
                "value_roi": closest_value["bbox"]
            }

            # Draw bounding box for key
            cv2.rectangle(image, (key_bbox[0], key_bbox[1]), 
                          (key_bbox[0] + key_bbox[2], key_bbox[1] + key_bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, key_text, (key_bbox[0], key_bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw bounding box for value
            value_x, value_y, value_w, value_h = closest_value["bbox"]
            cv2.rectangle(image, (value_x, value_y), 
                          (value_x + value_w, value_y + value_h), (255, 0, 0), 2)
            cv2.putText(image, closest_value["text"], (value_x, value_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Detected Text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# Print extracted key-value pairs with ROIs
print("Extracted Data:")
for key, data in extracted_data.items():
    print(f"{key}: {data['value']}, Key ROI: {data['key_roi']}, Value ROI: {data['value_roi']}")
