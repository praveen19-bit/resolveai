from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1:8b")

response = llm.invoke("Explain AI simply")
print(response)