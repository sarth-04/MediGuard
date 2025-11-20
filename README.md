# 🏥 MediGuard — Conversational Health RAG Bot

**MediGuard** is a modular, safety-focused Retrieval-Augmented Generation (RAG) chatbot designed for reliable and safe health-related assistance.  
It uses **Google Gemini** for reasoning and embeddings, **Streamlit** for the interface, and **Ragas** for evaluation.  
The system supports **Conversational Memory**, **Dynamic PDF Upload**, and **Dual Guardrails** for medical safety.

---

## ✨ Features

### 📄 Dynamic Knowledge Base
- Upload any **medical PDF** from the sidebar.
- Automatically re-indexes new documents.
- Falls back to default **diabetes management guidelines** if no PDF is provided.

### 🧠 Conversational Memory
- Maintains context across turns.
- Example:  
  - “What are the symptoms of diabetes?”  
  - “How do I treat it?” → Understands "it" refers to diabetes.

### 🛡️ Dual Guardrails
- **Input Guardrail:** Blocks non-medical or unsafe queries.
- **Output Guardrail:** Audits model responses for harmful or dangerous content before displaying.

### 📊 Ragas Evaluation
- Evaluate any response using:
  - **Faithfulness** → Checks grounding in the source document.
  - **Answer Relevancy** → Measures how well the answer fits the question.

### 💾 Persistent Vector Storage
- Uses **ChromaDB** with local persistence.
- Embeddings are reused across sessions → Saves API quota.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **LLM & Embeddings** | Google Gemini (gemini-2.0-flash, text-embedding-004) |
| **Orchestration** | LangChain + LCEL |
| **Vector Database** | ChromaDB |
| **Evaluation** | Ragas |
| **PDF Processing** | PyPDF |

---

## 🚀 Getting Started

### **Prerequisites**
- Python **3.10+**
- Google AI Studio API Key

### **Installation**

```bash
pip install -r requirements.txt
