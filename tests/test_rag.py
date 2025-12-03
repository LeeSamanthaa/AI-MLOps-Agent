# test_rag.py
import logging
logging.basicConfig(level=logging.INFO)

print("Testing RAG dependencies...")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("OK: langchain_community.embeddings")
except ImportError as e:
    print(f"FAILED: langchain_community - {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("OK: langchain_community.vectorstores")
except ImportError as e:
    print(f"FAILED: FAISS - {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("OK: sentence_transformers")
except ImportError as e:
    print(f"FAILED: sentence_transformers - {e}")

try:
    from modules.rag_system import initialize_rag_system, LANGCHAIN_AVAILABLE, FAISS_AVAILABLE
    print(f"RAG module: LANGCHAIN={LANGCHAIN_AVAILABLE}, FAISS={FAISS_AVAILABLE}")
    
    if LANGCHAIN_AVAILABLE and FAISS_AVAILABLE:
        print("Attempting initialization...")
        result = initialize_rag_system()
        print(f"Initialization result: {result}")
except Exception as e:
    print(f"FAILED: RAG module - {e}")

print("\nDone. Check logs for any errors.")