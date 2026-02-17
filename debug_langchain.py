import langchain
import os
print(f"Langchain version: {langchain.__version__}")
print(f"Langchain file: {langchain.__file__}")
print(f"Langchain types: {dir(langchain)}")

try:
    import langchain.chains
    print("langchain.chains imported successfully")
except ImportError as e:
    print(f"Error importing langchain.chains: {e}")

try:
    from langchain.chains import RetrievalQA
    print("RetrievalQA imported successfully")
except ImportError as e:
    print(f"Error importing RetrievalQA: {e}")
