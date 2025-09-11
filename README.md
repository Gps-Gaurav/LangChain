
# 🌟 LangChain

## 📌 What is LangChain?  
LangChain is an **open-source framework** designed to make it easier to build **applications powered by Large Language Models (LLMs)**.  

Instead of writing raw model calls every time, LangChain gives you:  
- 🔧 **Modular building blocks** (models, prompts, chains, memory, tools, agents)  
- 🔗 **Integrations** with external systems (databases, APIs, vector stores, web pages, PDFs)  
- 🧠 **High-level abstractions** to handle context, structured outputs, and reasoning  
- 🚀 **Production-ready features** like evaluation, monitoring, and scaling  

Think of LangChain as a **Swiss army knife for AI apps**.  

---

## 🚀 Why LangChain First?  
When building with LLMs, developers face challenges:  
- How do I **connect my app to data** (documents, databases, APIs)?  
- How do I manage **context, memory, and conversation history**?  
- How do I ensure **structured and reliable outputs**?  
- How do I combine multiple LLM calls in a workflow?  

LangChain solves these problems by offering:  
- **Model-agnostic development** → Use OpenAI, Anthropic, HuggingFace, Google, or open-source models.  
- **Chains & Memory** → Handle conversations, reasoning, and workflows.  
- **Tools & Agents** → Let your AI interact with external systems (search, APIs, calculators).  
- **Retrieval-Augmented Generation (RAG)** → Build smarter, grounded, knowledge-aware systems.  

---

## 🧩 Core Components of LangChain  

### 1️⃣ Models  
The heart of LangChain—how you talk to AI.  
- **LLMs** → General-purpose text generation.  
- **Chat Models** → Conversation-oriented (better than plain LLMs for dialogs).  
- **Embedding Models** → Turn text into vectors for semantic search & retrieval.  

👉 Example:  

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")
print(model.invoke("Tell me a joke about AI."))
```

---

### 2️⃣ Prompts  
Prompts guide the model’s output. LangChain makes them **dynamic, reusable, and structured**.  
- **Prompt Templates** → Insert variables (`{topic}`) into pre-written prompts.  
- **Chat Prompt Templates** → Multi-turn conversation templates.  
- **Message Placeholders** → Automatically inject conversation history.  

👉 Example:  

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a teacher."),
    ("user", "Explain {topic} like I’m 5.")
])

print(prompt.format(topic="machine learning"))
```

---

### 3️⃣ Chains  
Chains are **pipelines** where outputs of one step feed into the next.  
- **Simple Chain** → One prompt + one model.  
- **Sequential Chain** → Step-by-step execution.  
- **Parallel Chain** → Run multiple models at once.  
- **Conditional Chain** → Different logic paths (like if/else).  

👉 Example use case:  
1. Summarize a PDF → 2. Translate into Hindi → 3. Create a quiz.  

---

### 4️⃣ Memory  
LLMs are **stateless** (they forget after each call). LangChain adds memory:  
- **Buffer Memory** → Stores full conversation.  
- **Window Memory** → Keeps only last N messages.  
- **Summarizer Memory** → Compresses history into summaries.  
- **Custom Memory** → Store user preferences, facts, etc.  

This makes chatbots feel **personal and continuous**.  

---

### 5️⃣ Structured Output & Parsers  
By default, LLMs return text. But apps often need **JSON, CSV, or typed data**.  

LangChain offers:  
- **TypedDicts** → Define strict schemas.  
- **Pydantic Models** → Validate data structures.  
- **Output Parsers** → Convert raw text → structured formats.  

👉 Example: Extracting structured reviews:  

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

schema = [ResponseSchema(name="summary", type="string"),
          ResponseSchema(name="sentiment", type="string")]
parser = StructuredOutputParser.from_response_schemas(schema)

output = parser.parse('{"summary": "Great product", "sentiment": "positive"}')
print(output)
```

---

### 6️⃣ Runnables & LCEL (LangChain Expression Language)  
Instead of manually wiring components, LCEL lets you **compose workflows elegantly**.  

- **RunnableSequence** → Sequential steps  
- **RunnableParallel** → Run multiple tasks together  
- **RunnableLambda** → Custom Python functions in workflow  
- **RunnableBranch** → Conditional routing  

👉 Example (pseudocode):  

```python
chain = prompt | model | parser
response = chain.invoke({"input": "What is LangChain?"})
```

---

### 7️⃣ Document Loaders & Text Splitters  
Load data from **PDFs, websites, CSVs, text files, directories**.  
- `PyPDFLoader` → Extract text from PDFs.  
- `WebBaseLoader` → Scrape websites.  
- `CSVLoader` → Convert CSV rows into LangChain Documents.  

Large documents are split into **chunks** for efficient retrieval.  

---

### 8️⃣ Vector Stores & Retrievers  
Used for **semantic search** and **RAG**.  
- Vector Stores → FAISS, Pinecone, Chroma, Weaviate, Qdrant.  
- Retrievers → Search for relevant docs. Includes:  
  - **VectorStoreRetriever**  
  - **Multi-Query Retriever**  
  - **Contextual Compression Retriever**  

---

### 9️⃣ RAG (Retrieval-Augmented Generation)  
RAG = **LLM + knowledge base**.  

Steps:  
1. **Indexing** → Load → Chunk → Embed → Store in vector DB.  
2. **Retrieval** → Find top relevant chunks.  
3. **Augmentation** → Merge with user query.  
4. **Generation** → LLM produces grounded answers.  

👉 Benefits:  
- Uses **up-to-date data**  
- No model retraining needed  
- Handles **large document collections**  

---

### 🔟 Tools & Agents  
- **Tools** → Functions/APIs that an agent can call. (E.g., calculator, search, currency converter)  
- **Agents** → LLM-powered systems that **decide when & how to use tools**.  
- Popular Agent design → **ReAct (Reason + Act)** → LLM thinks step by step, calls tools, then answers.  

👉 Example:  
An agent that answers: *“What’s the population of Japan + 10%?”*  
1. Searches population (via API)  
2. Calls calculator tool  
3. Returns final answer  

---

## ⚡ Installation  

```bash
pip install langchain
pip install langchain-community   # integrations
pip install langchain-openai      # OpenAI support
pip install chromadb              # example vector store
```

---

## 📖 Quick Start Example  

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Model
model = ChatOpenAI(model="gpt-4o-mini")

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "{question}")
])

# Chain
chain = prompt | model
response = chain.invoke({"question": "Explain RAG in 2 lines."})

print(response)
```

---

## 🔗 Alternatives  
- [LlamaIndex](https://www.llamaindex.ai/) → Best for knowledge indexing  
- [Haystack](https://haystack.deepset.ai/) → Production search & RAG pipelines  

---

## 📚 References  
- [LangChain Docs](https://docs.langchain.com)  
- [LangSmith](https://smith.langchain.com) → Monitoring & Evaluation  

---

✨ **LangChain = Build AI apps faster.**  
With components for models, prompts, chains, memory, retrieval, and agents, you can go from **prototype → production** without reinventing the wheel.  
