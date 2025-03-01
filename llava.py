import base64
from gradio_client import Client, handle_file

client = Client("lintasmediadanawa/llava-test")
result = client.predict(
		prompt="Hello!!",
		image=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		api_name="/predict"
)
print(result)
