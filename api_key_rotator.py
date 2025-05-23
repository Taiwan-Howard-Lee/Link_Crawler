"""
API Key Rotator for managing multiple API keys and rotating between them.
This helps avoid rate limits by distributing requests across multiple keys.
"""
import os
import time
import logging
from typing import List, Optional, Dict, Any
import google.generativeai as genai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default API key environment variable names
DEFAULT_API_KEY_NAMES = ["GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3", "GEMINI_API_KEY4", "GEMINI_API_KEY5", "GEMINI_API_KEY6"]

class ApiKeyRotator:
    """
    A class to manage and rotate between multiple API keys.
    """
    def __init__(self, env_key_names: List[str] = None, cooldown_period: int = 60,
                max_uses_per_key: int = 10, recovery_period: int = 3600):
        """
        Initialize the API key rotator.

        Args:
            env_key_names: List of environment variable names containing API keys
                          (defaults to DEFAULT_API_KEY_NAMES)
            cooldown_period: Cooldown period in seconds after using a key
            max_uses_per_key: Maximum number of uses per key before considering it exhausted
            recovery_period: Time in seconds after which to check if an exhausted key has recovered
        """
        self.env_key_names = env_key_names or DEFAULT_API_KEY_NAMES
        self.cooldown_period = cooldown_period
        self.max_uses_per_key = max_uses_per_key
        self.recovery_period = recovery_period
        self.api_keys: List[str] = []
        self.key_usage: Dict[str, float] = {}  # Maps keys to last usage time
        self.key_use_count: Dict[str, int] = {}  # Maps keys to number of uses
        self.key_exhausted: Dict[str, bool] = {}  # Maps keys to exhaustion status
        self.key_exhausted_time: Dict[str, float] = {}  # Maps keys to time when they were exhausted
        self.current_key_index = 0

        # Load API keys from environment variables
        self._load_api_keys()

        if not self.api_keys:
            raise ValueError("No valid API keys found. Please check your environment variables.")

        logger.info(f"Initialized API Key Rotator with {len(self.api_keys)} keys")

    def _load_api_keys(self) -> None:
        """Load API keys from environment variables."""
        for key_name in self.env_key_names:
            key = os.getenv(key_name)
            if key and key.strip() and key != "your_gemini_api_key_here":
                self.api_keys.append(key)
                self.key_usage[key] = 0  # Initialize with no usage
                self.key_use_count[key] = 0  # Initialize with no uses
                self.key_exhausted[key] = False  # Initialize as not exhausted
                self.key_exhausted_time[key] = 0  # Initialize with no exhaustion time
            else:
                logger.warning(f"API key for {key_name} not found or invalid")

    def get_current_key(self) -> str:
        """Get the current API key."""
        if not self.api_keys:
            raise ValueError("No API keys available")
        return self.api_keys[self.current_key_index]

    def rotate_key(self) -> str:
        """
        Rotate to the next available API key.

        Returns:
            The next API key
        """
        if len(self.api_keys) <= 1:
            logger.warning("Only one API key available, cannot rotate")
            return self.get_current_key()

        # Move to the next key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")

        return self.get_current_key()

    def get_next_available_key(self) -> str:
        """
        Get the next available API key that is not in cooldown and not exhausted.

        Returns:
            An available API key

        Raises:
            RuntimeError: If all keys are exhausted
        """
        # Check if any exhausted keys have recovered
        self.check_all_keys_recovery()

        # Check if all keys are exhausted
        if self.all_keys_exhausted():
            raise RuntimeError("All API keys are exhausted. Cannot proceed with operation.")

        current_time = time.time()

        # Check all keys starting from the current one
        for _ in range(len(self.api_keys)):
            key = self.get_current_key()
            last_used = self.key_usage.get(key, 0)

            # Check if this key has recovered
            self.check_key_recovery(key)

            # Skip exhausted keys
            if self.key_exhausted.get(key, False):
                self.rotate_key()
                continue

            # If the key is available (not in cooldown)
            if current_time - last_used >= self.cooldown_period:
                logger.info(f"Using API key {self.current_key_index + 1}/{len(self.api_keys)} (uses: {self.key_use_count.get(key, 0)})")
                return key

            # Otherwise, rotate to the next key
            self.rotate_key()

        # If all non-exhausted keys are in cooldown, wait for the one with the shortest remaining cooldown
        min_wait = self.cooldown_period
        best_key = None

        for key, last_used in self.key_usage.items():
            # Check if this key has recovered
            if self.check_key_recovery(key):
                # If a key has recovered, use it immediately
                self.current_key_index = self.api_keys.index(key)
                logger.info(f"Using recovered API key {self.current_key_index + 1}/{len(self.api_keys)}")
                return key

            # Skip exhausted keys
            if self.key_exhausted.get(key, False):
                continue

            wait_time = self.cooldown_period - (current_time - last_used)
            if wait_time < min_wait:
                min_wait = wait_time
                best_key = key

        # If no non-exhausted key was found, raise an error
        if best_key is None:
            raise RuntimeError("All API keys are exhausted or in cooldown. Cannot proceed with operation.")

        # Set the current key to the best one
        self.current_key_index = self.api_keys.index(best_key)

        # Wait for the cooldown to expire
        if min_wait > 0:
            logger.info(f"All non-exhausted keys in cooldown. Waiting {min_wait:.2f} seconds...")
            time.sleep(min_wait)

        return best_key

    def mark_key_used(self, key: Optional[str] = None) -> None:
        """
        Mark an API key as used, updating its last usage time and count.

        Args:
            key: The API key to mark as used. If None, uses the current key.
        """
        if key is None:
            key = self.get_current_key()

        self.key_usage[key] = time.time()
        self.key_use_count[key] += 1

        # Check if the key has reached its maximum uses
        if self.key_use_count[key] >= self.max_uses_per_key:
            self.mark_key_exhausted(key)
            logger.warning(f"API key has reached maximum uses ({self.max_uses_per_key}) and is now marked as exhausted")

    def mark_key_exhausted(self, key: Optional[str] = None) -> None:
        """
        Mark an API key as exhausted.

        Args:
            key: The API key to mark as exhausted. If None, uses the current key.
        """
        if key is None:
            key = self.get_current_key()

        self.key_exhausted[key] = True
        self.key_exhausted_time[key] = time.time()
        logger.warning(f"API key marked as exhausted")

        # Check if all keys are exhausted
        if self.all_keys_exhausted():
            logger.error("All API keys are exhausted. Operations should stop.")

    def check_key_recovery(self, key: str) -> bool:
        """
        Check if an exhausted key has recovered based on the recovery period.

        Args:
            key: The API key to check

        Returns:
            True if the key has recovered, False otherwise
        """
        # If the key is not exhausted, no need to check recovery
        if not self.key_exhausted.get(key, False):
            return False

        current_time = time.time()
        exhausted_time = self.key_exhausted_time.get(key, 0)

        # Check if the recovery period has passed
        if current_time - exhausted_time >= self.recovery_period:
            logger.info(f"API key has potentially recovered after {self.recovery_period} seconds")
            # Reset the key's exhaustion status and use count
            self.key_exhausted[key] = False
            self.key_use_count[key] = 0
            return True

        return False

    def check_all_keys_recovery(self) -> None:
        """Check if any exhausted keys have recovered."""
        for key in self.api_keys:
            self.check_key_recovery(key)

    def all_keys_exhausted(self) -> bool:
        """
        Check if all API keys are exhausted.

        Returns:
            True if all keys are exhausted, False otherwise
        """
        # First check if any keys have recovered
        self.check_all_keys_recovery()

        return all(self.key_exhausted.get(key, False) for key in self.api_keys)

    def configure_gemini(self) -> bool:
        """
        Configure the Gemini API with the current key.

        Returns:
            True if configuration was successful, False if all keys are exhausted
        """
        try:
            key = self.get_next_available_key()
            genai.configure(api_key=key)
            self.mark_key_used(key)
            logger.info("Configured Gemini API with a fresh key")
            return True
        except RuntimeError as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            return False

    def get_gemini_model(self, model_name: str = 'gemini-2.0-flash') -> Optional[Any]:
        """
        Get a Gemini model instance with the current API key.

        Args:
            model_name: The name of the Gemini model to use
                       (default: 'gemini-2.0-flash' as specified by the user)

        Returns:
            A Gemini model instance, or None if all keys are exhausted
        """
        if not self.configure_gemini():
            logger.error("Could not get Gemini model: all API keys are exhausted")
            return None

        # List available models to help with debugging
        try:
            models = genai.list_models()
            model_names = [model.name for model in models]
            logger.info(f"Available Gemini models: {model_names}")
        except Exception as e:
            logger.warning(f"Could not list available models: {str(e)}")

        return genai.GenerativeModel(model_name)
