import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API keys for the trading bot
SOURCEGRAPH_API_TOKEN = os.getenv("SOURCEGRAPH_API_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")