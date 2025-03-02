from gradio_client import Client

client = Client("https://nickgambirasi-donut-text-extract.hf.space/")
result = client.predict(
				"image.png",	# str (filepath or URL to image) in 'image' Image component
				api_name="/predict"
)
print(result)


# 600 seconds per image processing
