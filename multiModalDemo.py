import os
import google.generativeai as genai
from dotenv import load_dotenv
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.generic_utils import (
    load_image_urls,
)
from PIL import Image


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)


def generate_code(img_url, prompt="Give tailwind CSS code for this image"):

    gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")
    image_documents = load_image_urls(img_url)

    stream_complete_response = gemini_pro.stream_complete(
        prompt=prompt,
        image_documents=image_documents,
    )

    for r in stream_complete_response:
        print(r.text, end="")


def main():

    url = ["https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"] #replace with image in the list
    generate_code(url,"What is this?")


if __name__ == "__main__":
    main()
