from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# HuggingFaceEndpoint will auto-pick HUGGINGFACEHUB_API_TOKEN
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

# Wrap with chat interface
model = ChatHuggingFace(llm=llm)

# Run query
result = model.invoke("What is the capital of India?")
print(result.content)
