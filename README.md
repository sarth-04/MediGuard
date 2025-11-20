🏥 MediGuard: Conversational Health RAG Bot

MediGuard is a robust, modular Retrieval-Augmented Generation (RAG) chatbot designed for health-related queries. It uses Google Gemini for reasoning and embeddings, Streamlit for the user interface, and Ragas for reference-free evaluation.

The system features Conversational Memory, allowing for multi-turn follow-up questions, and includes safety Guardrails to ensure the bot remains on-topic and provides safe medical information.

✨ Features

📄 Dynamic Knowledge Base: Upload your own medical PDFs via the sidebar to instantly update the bot's knowledge. Defaults to built-in diabetes management guidelines if no file is provided.

🧠 Conversational Memory: Remembers context from previous turns (e.g., "What are the symptoms?" -> "How do I treat it?").

🛡️ Dual Guardrails:

Input Rail: Blocks non-health-related queries (e.g., coding, politics).

Output Rail: Audits the AI's response for harmful or dangerous advice before displaying it.

📊 On-Demand Evaluation: Calculate Faithfulness and Answer Relevancy scores for specific responses using the Ragas framework powered by Gemini.

💾 Persistent Storage: Uses ChromaDB with local persistence to avoid re-embedding documents on every restart, saving API quota.

🛠️ Tech Stack

Frontend: Streamlit

LLM & Embeddings: Google Gemini (gemini-2.0-flash, text-embedding-004)

Orchestration: LangChain (LCEL)

Vector Database: ChromaDB

Evaluation: Ragas

PDF Processing: PyPDF

🚀 Getting Started

Prerequisites

Python 3.10+

Google AI Studio API Key: Get one here.

Installation

Clone the repository (or save the provided files to a folder).

Install dependencies:

pip install -r requirements.txt


Note: If you encounter issues with langchain, ensure you have the latest version:

pip install -U langchain langchain-community langchain-core langchain-google-genai langchain-chroma


Running the App

Execute the following command in your terminal:

streamlit run app.py


The application will launch in your default browser at http://localhost:8501.

📂 Project Structure

File

Description

app.py

Main Streamlit application. Handles UI, chat history, file uploads, and orchestrates the other modules.

rag_engine.py

Core RAG logic. Handles vector store creation, context retrieval, and conversational chain generation using Pure LCEL.

guardrails.py

Contains HealthGuardrail class for validating input safety and auditing output safety using LLM prompts.

evaluator.py

Ragas integration. Wraps the Gemini LLM to calculate performance metrics (Faithfulness, Relevancy).

data_loader.py

Manages PDF loading. Falls back to hardcoded text if no PDF is found.

config.py

Central configuration for API keys, model names, and paths.

requirements.txt

List of Python dependencies.

📖 Usage Guide

Enter API Key: Paste your Google Gemini API key in the sidebar.

Upload Data (Optional): Use the "Upload Medical PDF" button in the sidebar. The system will index the new file immediately.

Chat: Ask health questions.

Example 1: "What are the symptoms of Type 2 Diabetes?"

Example 2: "What is the treatment protocol for it?" (Testing memory)

Evaluate: If you want to check the quality of the last response, click "📊 Evaluate Last Response" in the sidebar.

Faithfulness: Measures if the answer is derived only from the provided document (avoids hallucinations).

Relevancy: Measures how pertinent the answer is to your question.

⚠️ Note on Quota

This app uses ChromaDB with persistence (./chroma_db folder). This means it only calls the Google Embedding API when you upload a new file or run the app for the first time. This prevents you from hitting Google's free tier Rate Limits (429 Errors) during development.
