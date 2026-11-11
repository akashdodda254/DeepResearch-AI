from agents.llm_client import LocalLLM

def main():
    llm = LocalLLM()

    question = (
        "Explain Retrieval-Augmented Generation in the context "
        "of academic research in three sentences."
    )

    print("Connecting to local LLM...")

    answer = llm.generate(question)

    print("\nMODEL RESPONSE")
    print("--------------")
    print(answer)

    assert len(answer) > 20

    print("\nLLM CONNECTION TEST PASSED!")

if __name__ == "__main__":
    main()