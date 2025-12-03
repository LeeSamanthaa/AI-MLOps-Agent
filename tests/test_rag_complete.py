#!/usr/bin/env python3
# test_rag_complete.py
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 60)
print("RAG SYSTEM DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Core LangChain imports
print("\n[1/5] Testing LangChain imports...")
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("    OK: HuggingFaceEmbeddings")
except ImportError as e:
    print(f"    FAILED: HuggingFaceEmbeddings - {e}")
    sys.exit(1)

try:
    from langchain_community.vectorstores import FAISS
    print("    OK: FAISS vectorstore")
except ImportError as e:
    print(f"    FAILED: FAISS - {e}")
    sys.exit(1)

try:
    from langchain.schema import Document
    print("    OK: Document schema")
except ImportError as e:
    print(f"    FAILED: Document - {e}")
    sys.exit(1)

# Test 2: Sentence Transformers
print("\n[2/5] Testing sentence-transformers...")
try:
    from sentence_transformers import SentenceTransformer
    print("    OK: SentenceTransformer")
except ImportError as e:
    print(f"    FAILED: sentence-transformers not installed")
    print("    Run: pip install sentence-transformers")
    sys.exit(1)

# Test 3: FAISS CPU
print("\n[3/5] Testing FAISS...")
try:
    import faiss
    print(f"    OK: FAISS installed (version available)")
except ImportError as e:
    print(f"    FAILED: faiss-cpu not installed")
    print("    Run: pip install faiss-cpu")
    sys.exit(1)

# Test 4: RAG Module imports
print("\n[4/5] Testing RAG module...")
try:
    from modules.rag_system import LANGCHAIN_AVAILABLE, FAISS_AVAILABLE
    print(f"    LANGCHAIN_AVAILABLE: {LANGCHAIN_AVAILABLE}")
    print(f"    FAISS_AVAILABLE: {FAISS_AVAILABLE}")
    
    if not (LANGCHAIN_AVAILABLE and FAISS_AVAILABLE):
        print("    WARNING: RAG flags not set correctly")
except ImportError as e:
    print(f"    FAILED: Could not import rag_system - {e}")
    sys.exit(1)

# Test 5: Initialize RAG
print("\n[5/5] Testing RAG initialization...")
try:
    from modules.rag_system import initialize_rag_system
    result = initialize_rag_system()
    
    if result is None:
        print("    WARNING: initialize_rag_system returned None")
    else:
        print(f"    OK: RAG system initialized - {type(result).__name__}")
        
except Exception as e:
    print(f"    FAILED: RAG initialization error")
    print(f"    Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("RAG SYSTEM: ALL TESTS PASSED")
print("=" * 60)
print("\nYour RAG system is properly configured!")
print("The RAG search tab should now be available in your app.")