from gradio_client import Client, handle_file
import time

start_time = time.time()

client = Client("p2kalita/Donut")
result = client.predict(
		image=handle_file('image.png'),
		api_name="/run_prediction"
)
print(result)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# 20 - 30 seconds per image processing
