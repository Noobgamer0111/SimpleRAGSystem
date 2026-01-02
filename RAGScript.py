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

client = lms.get_default_client()
model1 = client.llm.load_new_instance("bartowski/Llama-3.2-1B-Instruct-GGUF", ttl =3600)
model2 = client.embedding_model.load_new_instance("CompendiumLabs/bge-base-en-v1.5-gguf", ttl=3600)

# 2) Create a knowledge base.
# For this example, we will use this existing knowledge base on cat facts.
# Download here: https://huggingface.co/ngxson/demo_simple_rag_py/blob/main/cat-facts.txt