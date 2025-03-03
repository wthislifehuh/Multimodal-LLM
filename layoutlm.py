from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

# Load model & processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Load document image
image = Image.open("./assets/waybill.png").convert("RGB")

# Process image for model
encoding = processor(image, return_tensors="pt")

# Predict text and layout positions
outputs = model(**encoding)

# import torch
# from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
# from PIL import Image

# # Load model and processor
# processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
# model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# # Move model to CPU
# device = "cpu"
# model.to(device)

# # Load document image
# image = Image.open("./assets/waybill.png").convert("RGB")

# # Process image
# encoding = processor(image, return_tensors="pt")
# encoding = {k: v.to(device) for k, v in encoding.items()}  # Ensure data is on CPU

# # Predict layout
# with torch.no_grad():
#     outputs = model(**encoding)
