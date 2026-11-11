import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()


class GeminiLLM:
    """Client for generating text using Google's Gemini API."""

    def __init__(self, model="gemini-3.5-flash-lite"):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your .env file."
            )

        self.client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 30000}
        )

        self.model = model

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text with limited retries for temporary API errors."""

        full_prompt = prompt

        if system_prompt:
            full_prompt = (
                f"System instructions:\n{system_prompt}\n\n"
                f"User request:\n{prompt}"
            )

        for attempt in range(3):

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text.strip()

            except errors.ServerError as error:

                if attempt == 2:
                    raise RuntimeError(
                        f"Gemini server unavailable after 3 attempts: {error}"
                    ) from error

                wait_time = 5 * (2 ** attempt)

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            except Exception as error:
                raise RuntimeError(
                    f"Gemini API generation failed: {error}"
                ) from error

        raise RuntimeError("Gemini generation failed.")