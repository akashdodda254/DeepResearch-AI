import unittest
import os

from unittest.mock import patch, MagicMock

from agents.llm_client import GeminiLLM


class TestGeminiLLM(unittest.TestCase):

    @patch("agents.llm_client.genai.Client")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key"})
    def test_generate(self, mock_client):

        mock_response = MagicMock()
        mock_response.text = "Test Gemini response"

        mock_client.return_value.models.generate_content.return_value = (
            mock_response
        )

        llm = GeminiLLM()

        result = llm.generate("Test prompt")

        self.assertEqual(result, "Test Gemini response")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):

        with self.assertRaises(ValueError):
            GeminiLLM()


if __name__ == "__main__":
    unittest.main()