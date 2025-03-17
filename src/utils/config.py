import os
from dotenv import load_dotenv

load_dotenv()

SOURCEGRAPH_API_TOKEN = os.getenv("SOURCEGRAPH_API_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")