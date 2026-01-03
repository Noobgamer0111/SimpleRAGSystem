# Simple RAG Script
# This script demonstrates a basic implementation of a Retrieval-Augmented Generation (RAG) 
# model.
# It retrieves relevant documents from a knowledge base and generates responses based on them.

# Original Tutorial: https://medium.com/@anish.chitturu/building-your-own-rag-system-from-scratch-a-step-by-step-guide-7186fcbb3b14

# 1) Import the models into a LM backend.
# We will use the models specified in the tutorial.
# Models used:
# https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf
# https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF

# For the this example, I will use LM Studio as the LM backend.
# Documentation: https://lmstudio.ai/docs/python

import lmstudio as lms
import wget
import os
import numpy as np

client = lms.get_default_client()
model1 = client.llm.load_new_instance("bartowski/Llama-3.2-1B-Instruct-GGUF", ttl =3600)
model2 = client.embedding_model.load_new_instance("CompendiumLabs/bge-base-en-v1.5-gguf", ttl=3600)

# 2) Create a knowledge base.
# For this example, we will use this existing knowledge base using LM Studio on cat facts.
# Download here: https://huggingface.co/ngxson/demo_simple_rag_py/blob/main/cat-facts.txt

# Download the knowledge base file for the first time if it does not exist.
file_path = "cat-facts.txt"
if not os.path.exists(file_path):
    print(f"Downloading knowledge base from HuggingFace...")
    url = "https://huggingface.co/ngxson/demo_simple_rag_py/raw/main/cat-facts.txt"
    response = wget.download(url)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Knowledge base downloaded and saved to {file_path}")
else:
    print(f"Knowledge base file already exists at {file_path}")

# 3) Load the knowledge base and create embeddings for each document.

with open("cat-facts.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n")
embeddings = []
for doc in documents:
    model = lms.embedding_model(model2)
    embedding = model.get_embedding(doc)
    embeddings.append((doc, embedding))

print(f"Loaded {len(embeddings)} documents into the knowledge base.")

# 4) Define a function to define cosine similarity between two vectors.
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = sum([x ** 2 for x in vec1]) ** 0.5
    norm_vec2 = sum([x ** 2 for x in vec2]) ** 0.5
    return dot_product / (norm_vec1 * norm_vec2)

# 5) Define a function to retrieve the most relevant documents based on a query.
def retrieve(query, top_n=3):
    model = lms.embedding_model(model2)
    query_embedding = model.get_embedding(query)
    similarities = []
    for doc, emb in embeddings:
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((doc, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, sim in similarities[:top_n]]

# 6) Let's take a user query, retrieve relevant documents, and generate a response.

input_query = input('Ask me a question about cats: ')
retrieved_docs = retrieve(input_query)

print ("Retrieved Knowledge:")
for chunk, similarity in retrieved_docs:
    print(f"- (similarity: {similarity:.2f}) {chunk}")

# 7) Generate a response in LM Studio using the retrieved documents as context.

for fragment in model.respond_stream(input_query):
    print(fragment.content, end='', flush=True)