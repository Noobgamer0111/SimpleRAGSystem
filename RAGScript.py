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
import requests
import os
import numpy as np
import argparse
import logging
import sys

# Keep placeholders for cleanup; actual model instances are created in main()
llm = None
embed_model = None
client = None

# (Knowledge-base loading and embedding creation are handled inside main())

# 4) Define a function to define cosine similarity between two vectors.
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

# 5) Define a function to retrieve the most relevant documents based on a query.
def retrieve(query, embeddings, embed_model, top_n=3):
    """Retrieve top_n documents given a query, embeddings list and an embedding model.

    embeddings: list of (doc_text, np.array(vector))
    embed_model: object with method `embed_query(text)` returning a vector-like sequence
    """
    query_embedding = np.array(embed_model.embed_query(query))
    similarities = []
    for doc, emb in embeddings:
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((doc, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# 6) Let's take a user query, retrieve relevant documents, and generate a response.

def main():
    parser = argparse.ArgumentParser(description="Simple RAG demo using LM Studio")
    parser.add_argument("--file-path", default="cat-facts.txt", help="path to knowledge base text file")
    parser.add_argument("--top-n", type=int, default=3, help="number of documents to retrieve")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    global client, llm, embed_model
    client = lms.get_default_client()
    # Load LLM and embedding model instances with error handling
    try:
        logger.info("Loading models into LM Studio client...")
        llm = client.llm.load_new_instance("bartowski/Llama-3.2-1B-Instruct-GGUF", ttl=3600)
        embed_model = client.embedding_model.load_new_instance("CompendiumLabs/bge-base-en-v1.5-gguf", ttl=3600)
        logger.info("Models loaded")
    except Exception:
        logger.exception("Failed to load models from LM Studio")
        sys.exit(1)

    # 2) Create a knowledge base.
    file_path = args.file_path
    if not os.path.exists(file_path):
        logger.info("Downloading knowledge base from HuggingFace...")
        url = "https://huggingface.co/ngxson/demo_simple_rag_py/raw/main/cat-facts.txt"
        resp = requests.get(url)
        try:
            resp.raise_for_status()
        except Exception:
            logger.exception("Failed to download knowledge base from %s", url)
            sys.exit(1)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        logger.info("Knowledge base downloaded and saved to %s", file_path)
    else:
        logger.info("Knowledge base file already exists at %s", file_path)

    # 3) Load the knowledge base and create embeddings for each document.
    with open(file_path, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines() if line.strip()]
    embeddings = []
    for doc in documents:
        try:
            emb = embed_model.embed_query(doc)
        except Exception:
            logger.exception("Failed to create embedding for a document")
            emb = [0.0]
        embeddings.append((doc, np.array(emb)))

    logger.info("Loaded %d documents into the knowledge base.", len(embeddings))

    input_query = input('Ask me a question about cats: ')
    retrieved_docs = retrieve(input_query, embeddings, embed_model, top_n=args.top_n)

    print("Retrieved Knowledge:")
    for chunk, similarity in retrieved_docs:
        print(f"- (similarity: {similarity:.2f}) {chunk}")

    # 7) Generate a response in LM Studio using the retrieved documents as context.
    context = "\n\n".join([chunk for chunk, _ in retrieved_docs])
    prompt = f"You are a helpful assistant. Use the context below to answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {input_query}\nAnswer:"

    # Use the LLM instance to generate a response (stream or single response)
    try:
        # prefer streaming if available
        for fragment in llm.respond_stream(prompt):
            print(fragment.content, end='', flush=True)
    except Exception:
        logger.debug("Streaming not available or failed, falling back to non-streaming respond", exc_info=True)
        try:
            result = llm.respond(prompt)
            print(result.output_text if hasattr(result, 'output_text') else str(result))
        except Exception:
            logger.exception("LLM generation failed")


if __name__ == '__main__':
    try:
        main()
    finally:
        # Attempt cleanup of loaded model instances using common method names if present
        for inst in (llm, embed_model):
            if inst is None:
                continue
            for cleanup_name in ("unload_instance", "unload", "close", "destroy", "shutdown"):
                fn = getattr(inst, cleanup_name, None)
                if callable(fn):
                    try:
                        fn()
                        logger.info("Cleaned up model instance via %s", cleanup_name)
                        break
                    except Exception:
                        logger.debug("Cleanup method %s failed", cleanup_name, exc_info=True)