from langchain_ollama import ChatOllama

def main():

    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0
    )

    response = llm.invoke("Explain what Kafka is in two sentences.")

    print("\nLLM Response:")
    print(response.content)

if __name__ == "__main__":
    main()