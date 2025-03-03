import base64
from gradio_client import Client, handle_file
import time

start_time = time.time()

client = Client("lintasmediadanawa/llava-test")
result = client.predict(
		prompt="What is in the image?",
		image=handle_file('./image.png'),
		api_name="/predict"
)
print(result)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# 80 - 90 seconds per image processing
