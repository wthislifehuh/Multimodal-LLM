import base64
from gradio_client import Client, handle_file
import time

start_time = time.time()

client = Client("lintasmediadanawa/llava-test")
result = client.predict(
		prompt="Hello!!",
		image=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		api_name="/predict"
)
print(result)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
