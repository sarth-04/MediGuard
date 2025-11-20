import os
import streamlit as st

# Global Constants
# Switched to the newer, more efficient model
EMBEDDING_MODEL = "models/text-embedding-004" 
CHAT_MODEL = "gemini-2.0-flash"
VECTOR_STORE_PATH = "./chroma_db"

def setup_env(api_key: str):
    """Sets up the environment variables required for the libraries."""
    if not api_key:
        st.error("API Key is missing!")
        st.stop()
    os.environ["GOOGLE_API_KEY"] = api_key