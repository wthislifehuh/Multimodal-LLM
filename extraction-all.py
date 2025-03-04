import time
import json
import requests
import base64
import re
import cv2
import numpy as np
from paddleocr import PaddleOCR

class ParcelDataExtractor:
    def __init__(self, api_key, model_name="gpt-4o-2024-08-06"):
        self.api_key = api_key
        self.model_name = model_name

    def encode_image(self, image_path):
        """Encodes image to base64 format."""
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
        print("Extracted Data:", extracted_data)
        return extracted_data


    def openAI_analysis(self, extracted_data):
        """Uses GPT-4o API to identify key-value pairs and merge their coordinates."""
        prompt = f"""
        Identify key-value pairs from the given OCR extracted text and ROIs. 
        Match the most likely values to their corresponding keys and merge both the key and value ROIs to form a larger bounding box.
        Only return in JSON format, no other text or comments.
        Example JSON format:
        [
            {{"key": "ACCOUNT", "value": "1234567", "combined_roi": "(62, 217, 250, 30)"}}
        ]

        Data: {json.dumps(extracted_data)}
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert in text extraction and document processing."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
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

        for item in analyzed_data:
            key = item["key"]
            value = item["value"]
            combined_roi = item["combined_roi"]

            # Convert the combined_roi from string to tuple (x, y, w, h)
            x, y, w, h = map(int, combined_roi.strip("()").split(", "))

            # Draw bounding box around the detected key-value pair
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put label text
            label = f"{key}: {value}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the image with bounding boxes
        cv2.imshow("Annotated Waybill", image)
        cv2.waitKey(0)  # Wait for key press to close window
        cv2.destroyAllWindows()

    def process_image(self, image_path):
        """Main function to extract text and ROIs, then analyze using GPT-4o."""
        extracted_data = self.extract_text_rois(image_path)
        analyzed_data, processing_time = self.openAI_analysis(extracted_data)

        # Visualize results using OpenCV
        self.visualize_results(image_path, analyzed_data)

        return analyzed_data, processing_time


# Example Usage
api_key = "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"
image_path = "./assets/waybill.png"
model_name = "gpt-4o-2024-08-06"


start_time = time.time()
extractor = ParcelDataExtractor(api_key)
output, time_taken = extractor.process_image(image_path)

print("Final Output JSON:")
print(json.dumps(output, indent=4))
print(f"Processing Time: {time_taken:.2f} seconds")
end_time = time.time()
print(f"Total Time Taken: {end_time - start_time:.2f} seconds")