from astrapy.db import AstraDB
from dotenv import load_dotenv
import os

load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")



# Initialization
db = AstraDB(
  token=ASTRA_DB_APPLICATION_TOKEN,
  api_endpoint=ASTRA_DB_API_ENDPOINT,
  )


db.delete_collection("tailwind_embeddings")
print(f"Connected to Astra DB: {db.get_collections()}")