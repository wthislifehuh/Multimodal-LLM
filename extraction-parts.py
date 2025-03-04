import time
import json
import requests
import base64
import re
import cv2
import numpy as np
import os
from paddleocr import PaddleOCR

class ParcelDataExtractor:
    def __init__(self, api_key, model_name="gpt-4o-2024-08-06"):
        self.api_key = api_key
        self.model_name = model_name

    def encode_image(self, image_path):
        """Encodes an image in base64 format for API request."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract_text_rois(self, image_path):
        """Extracts text and bounding box coordinates from an image using PaddleOCR."""
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        results = ocr.ocr(image_path, cls=True)

        extracted_data = []
        for result in results[0]:
            bbox, text, prob = result[0], result[1][0], result[1][1]
            x, y, w, h = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0] - bbox[0][0]), int(bbox[2][1] - bbox[0][1])
            extracted_data.append({"text": text, "roi": (x, y, w, h)})

        return extracted_data

    def openAI_analysis(self, extracted_data, image_base64):
        """Uses GPT-4o API to identify key-value pairs and merge their coordinates using both text and image analysis."""
        prompt = f"""
        Given an image, OCR-extracted text and its bounding box ROI coordinates, accurately identify key-value pairs, match the most likely values to their corresponding keys.
        Return only the key and coordinate of the value in JSON format.

        Example response:
        [
            {{"key": "ACCOUNT", "roi": "(78, 289, 250, 30)"}},
        ]

        OCR-Extracted Text Data: {json.dumps(extracted_data)}
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert in document OCR analysis."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]}
            ],
            "temperature": 0,
            "max_tokens": 500
        }

        start_time = time.time()
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        end_time = time.time()
        processing_time = end_time - start_time

        if response.status_code == 200:
            response_json = response.json()
            try:
                content = response_json['choices'][0]['message']['content']
                if not content.strip():
                    raise ValueError("Empty content received from API")

                match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    content_json_str = match.group(1).strip()
                else:
                    raise ValueError("Invalid JSON format received from API")

                output_json = json.loads(content_json_str)
                return output_json, processing_time
            except json.JSONDecodeError as e:
                raise ValueError(f"An error occurred while decoding the JSON response: {e}")
        else:
            raise ConnectionError(f"Error in API request: {response.status_code}, details: {response.text}")

    def visualize_results(self, image_path, analyzed_data):
        """Displays image with bounding boxes for extracted key-value pairs using OpenCV."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        for item in analyzed_data:
            key = item["key"]
            roi = item["roi"]

            # Ensure the roi is formatted correctly as (x, y, w, h)
            roi_clean = re.sub(r'[^0-9,]', '', roi) 

            try:
                x, y, w, h = map(int, roi_clean.split(","))
            except ValueError:
                print(f"Skipping invalid ROI format: {roi}")
                continue  # Skip this entry if it has invalid coordinates

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put key label
            label = f"{key}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the annotated image
        cv2.imshow("Annotated Waybill", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def process_image(self, image_path):
        """Main function to extract text and ROIs, then analyze using GPT-4o."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        extracted_data = self.extract_text_rois(image_path)
        if not extracted_data:
            raise ValueError("No text was extracted from the image")
        
        image_base64 = self.encode_image(image_path)
        analyzed_data, processing_time = self.openAI_analysis(extracted_data, image_base64)

        # Visualize results using OpenCV
        self.visualize_results(image_path, analyzed_data)

        return analyzed_data, processing_time


# Example Usage
api_key = ""
image_path = "./assets/waybill2.png"
model_name = "gpt-4o-2024-08-06"

start_time = time.time()
extractor = ParcelDataExtractor(api_key)
output, time_taken = extractor.process_image(image_path)

print("Final Output JSON:")
print(json.dumps(output, indent=4))
print(f"Processing Time: {time_taken:.2f} seconds")
end_time = time.time()
print(f"Total Time Taken: {end_time - start_time:.2f} seconds")
