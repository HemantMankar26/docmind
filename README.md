# 🧠 DocMind — AI Document Chatbot

> Upload any PDF. Ask anything. Get answers grounded in your documents — with source citations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=flat-square&logo=streamlit)

---

## 📌 What is DocMind?

DocMind is an end-to-end **Retrieval Augmented Generation (RAG)** application that allows users to:

- Upload one or multiple PDF documents
- Ask natural language questions about the content
- Receive accurate, context-grounded answers with **source citations**
- Maintain **conversation memory** across multiple questions

Built as a portfolio project to demonstrate real-world AI engineering skills — RAG pipeline design, vector search, LLM integration, and production-ready UI.

---

## 🏗️ Architecture

```
User uploads PDF(s)
        ↓
 PyPDF loads pages
        ↓
 RecursiveTextSplitter → chunks (1000 tokens, 150 overlap)
        ↓
 HuggingFace Embeddings (all-MiniLM-L6-v2) → vectors
        ↓
 FAISS VectorStore (saved locally)
        ↓
 User asks question
        ↓
 Semantic similarity search → top 4 relevant chunks
        ↓
 Groq LLaMA3 LLM generates grounded answer
        ↓
 Answer + source citations displayed in Streamlit UI
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq API — LLaMA3-8B (free, fast) |
| RAG Framework | LangChain |
| Vector Database | FAISS (local) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| PDF Parsing | PyPDF |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/HemantMankar26/docmind.git
cd docmind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a FREE Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up (free)
- Create an API key

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Use DocMind
- Paste your Groq API key in the sidebar
- Upload one or more PDF files
- Click **Process Documents**
- Start asking questions!

---

## 📁 Project Structure

```
docmind/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore
├── .streamlit/
│   └── config.toml         # Streamlit theme config
└── src/
    ├── __init__.py
    ├── loader.py            # PDF loading and text chunking
    ├── embedder.py          # HuggingFace embeddings + FAISS vectorstore
    └── chain.py             # LangChain RAG chain with memory
```

---

## ✨ Features

- ✅ Multi-PDF support — upload and query multiple documents simultaneously
- ✅ Conversational memory — remembers previous questions in the session
- ✅ Source citations — shows which document and page each answer came from
- ✅ Dark mode UI — clean, professional interface
- ✅ Fully local embeddings — no embedding API costs
- ✅ Fast inference — Groq's LPU delivers sub-second LLM responses

---

## 🔮 Future Improvements

- [ ] Support for DOCX, TXT, and CSV files
- [ ] ChromaDB persistent vectorstore for cross-session memory
- [ ] Export chat history as PDF report
- [ ] Docker containerization
- [ ] Authentication for multi-user deployment

---

## 👤 Author

**Hemant Mankar**
AI/ML Engineer | Computer Vision • NLP • LLMs

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/hemant-mankar-3204a7225)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/HemantMankar26)

---

## 📄 License

MIT License — free to use, modify, and distribute.
