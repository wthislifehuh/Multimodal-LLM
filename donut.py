from gradio_client import Client, handle_file

client = Client("p2kalita/Donut")
result = client.predict(
		image=handle_file('image.png'),
		api_name="/run_prediction"
)
print(result)


# 20 - 30 seconds per image processing
