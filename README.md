# Retrieval-Augmented Generation (RAG) Project

This project implements a **Retrieval-Augmented Generation (RAG)** system that combines a language model with vector database retrieval to provide more accurate and context-aware responses.  
It uses **Pinecone** as a vector store and integrates with **GROQ API** for LLM-powered reasoning.

---

## 🚀 Features

- **RAG Pipeline**: Enhances answers with retrieved knowledge from a vector database.  
- **Pinecone Integration**: Stores and retrieves embeddings efficiently.  
- **SQLite Support**: Local database (`1_multistep_rag.sqlite`) for multi-step workflows.  
- **Environment-Based Config**: API keys and configs managed via `.env`.  
- **Extensible**: Easily adaptable for different datasets and tasks.  

---

## 🛠️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/keshabshrestha007/2_Langgraph_RAG_Agent.git
```
```bash
cd 2_Langgraph_RAG_Agent
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
```
On Linux/Mac
```bash
source venv/bin/activate
```
On Windows
```bash
venv\Scripts\activate       
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key.

```bash
copy .env.example .env
# edit .env to set GROQ_API_KEY (no surrounding quotes preferred)
```
### 5. Run the Streamlit app:

```bash
streamlit run app.py
```
---

## 📂 Project Structure

```
.
├── .env                 # Environment variables (API keys, configs)
├── .gitignore           # Ignored files & folders
├── 1_multistep_rag.sqlite  # SQLite database for RAG state
├── main.py              # Main entrypoint (example)
├── requirements.txt     # Python dependencies
├── modules/             # Custom modules (retriever, LLM wrapper, etc.)
└── README.md            # Project documentation
```
---
## ⚙️ Development Notes
Temporary files, cache, and local environment are ignored via .gitignore.

SQLite database (1_multistep_rag.sqlite) supports multi-step conversation memory.

API keys must not be committed to GitHub.

---

## 📌 Future Improvements
Add support for additional vector stores (FAISS, Weaviate).

Improve multi-step reasoning with agentic workflows.

Build a simple Streamlit UI for testing queries.

---

## 📝 License
This project is open-source. Use at your own risk, and keep your API keys safe.



