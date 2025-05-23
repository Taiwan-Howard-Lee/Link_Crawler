"""
Configuration settings for the link crawler application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Crawler settings
REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# CSV settings
DEFAULT_URL_COLUMN = "url"
DEFAULT_SUMMARY_COLUMN = "summary"
