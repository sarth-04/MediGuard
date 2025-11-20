import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Default fallback text if no PDF is provided
DEFAULT_TEXT = """
DIABETES MANAGEMENT GUIDELINES 2024 (FALLBACK DATA)

1. Overview
Diabetes mellitus is a chronic condition characterized by high blood glucose levels. 
Type 1 diabetes is an autoimmune reaction where the body attacks insulin-producing cells. 
Type 2 diabetes is characterized by insulin resistance.

2. Symptoms
Common symptoms include frequent urination (polyuria), excessive thirst (polydipsia), 
extreme hunger (polyphagia), unexplained weight loss, and blurred vision.
"""

def get_documents(file_path=None):
    """
    Logic:
    1. If file_path is provided and exists, load that specific PDF.
    2. If no file_path, return the hardcoded dummy text.
    """
    if file_path and os.path.exists(file_path):
        print(f"📄 Loading user document from: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()
    else:
        print("⚠️ No PDF provided. Using default dummy text.")
        return [Document(page_content=DEFAULT_TEXT, metadata={"source": "dummy_guidelines"})]