# 🩺 docBot

**docBot** is an AI-powered chatbot that answers medical queries based on content extracted from user-uploaded PDF documents. It uses vector embeddings and a custom LLM prompt to provide accurate and context-specific responses.

## 🚀 Features

- Uses Hugging Face sentence-transformer embeddings for vectorization.
- Employs FAISS as a vector store for efficient document retrieval.
- Integrates with the Mistral-7B-Instruct model via Hugging Face Inference Endpoint.
- Streamlit chat interface for easy user interaction.
- Caches vectorstore for performance optimization.

## 📦 Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Transformers & Inference Endpoints

## 🛠️ Setup Instructions

### ✅ Steps to Run docBot Locally

1. **Install Pipenv** (if not already installed):
   ```bash
   pip install pipenv
2. **Create and activate a virtual environment:**
   ```bash
   pipenv shell
3. **Install dependencies:**
   ```bash
   pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf streamlit
4. ****Generate a Hugging Face token** from [https://huggingface.co/](https://huggingface.co/), then create a `.env` file in your project root and add:**
   ```bash
   HF_TOKEN=your_huggingface_token
5. **Ensure the FAISS vector store is present at:**
   ```bash
   vectorstore/db_faiss/
5. **Run the application:**
   ```bash
   streamlit run docBot.py