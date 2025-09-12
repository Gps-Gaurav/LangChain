from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Narendra Modi is the Prime Minister of India known for his leadership and economic reforms.",
    "Rahul Gandhi is a prominent leader of the Indian National Congress with a strong political legacy.",
    "Dr. B. R. Ambedkar, the principal architect of the Indian Constitution, was a social reformer and politician.",
    "Atal Bihari Vajpayee was a former Prime Minister of India, admired for his oratory skills and statesmanship.",
    "Arvind Kejriwal is the Chief Minister of Delhi, known for his focus on education and healthcare reforms."
]

query = "tell me about Arvind Kejriwal"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)