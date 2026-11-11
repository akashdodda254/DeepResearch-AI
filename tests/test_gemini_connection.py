from agents.llm_client import GeminiLLM

def main():
    print("Connecting to Gemini API...")

    llm = GeminiLLM()

    response = llm.generate(
        prompt="Explain Retrieval-Augmented Generation in exactly two sentences.",
        system_prompt="You are a concise AI research assistant."
    )

    print("\nGemini response:")
    print(response)

    print("\nGemini API connection successful!")

if __name__ == "__main__":
    main()