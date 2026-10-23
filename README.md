# 🏛️ Orient BlackSwan History Bot (Project 4)

A full-stack Retrieval-Augmented Generation (RAG) application built using Python, LangChain, ChromaDB, FastAPI, and a modern glassmorphism frontend UI. This system queries the *Orient BlackSwan History catalogue (2024–25)*, provides precise answers with source page references, and enforces strict guardrails by replying "Not found" when information is absent.

---

## 🚀 System Architecture & Flow

The project is structured into a robust 3-tier pipeline:

1. **Data Ingestion (`ingest.py`):**
   * Loads the PDF document (`History_2024-25.pdf`) using LangChain's `PyPDFLoader`.
   * Splits text content into manageable chunks using `RecursiveCharacterTextSplitter`.
   * Computes dense vector embeddings and persists them locally into a Chroma vector database (`history_db`).

2. **Backend Server (`server.py`):**
   * Built with **FastAPI** and initialized using an asynchronous lifespan context manager to load the RAG chain on startup.
   * Utilizes **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) and an LCEL generation chain with an LLM.
   * Implements a retriever fetching the top 4 relevant chunks and exposes a POST `/ask` endpoint with CORS enabled.

3. **Frontend UI (`index.html`):**
   * A responsive, single-file glassmorphism chatbot interface styled with custom CSS and JavaScript.
   * Sends asynchronous POST requests to the FastAPI backend and smoothly renders answers, source citations, and collapsible retrieved context chunks.

---

## 📋 The 5 Catalogue Questions & Answers

| Q# | Question | Answer Generated | Source Reference |
|---|---|---|---|
| **1** | On what date were the prices in this catalogue last updated? | The prices were last updated on 20–01–2025. | Page 24 and Page 91 |
| **2** | In which year was the publisher (Orient BlackSwan) incorporated? | Orient BlackSwan was incorporated in 1948. | Page 2 |
| **3** | What is the price of a Physics textbook? | Not found | Out-of-domain query handled via guardrails |
| **4** | Who is the current Minister of Education? | Not found | Out-of-domain query handled via guardrails |
| **5** | Which is the "best" history book for a beginner? | Not found | Subjective query handled via strict context grounding |

---

## 📸 Screenshots & Execution Results

### 1. Factual Lookups (Questions 1 & 2)
*Successfully retrieved exact catalog updates and corporate history with correct page citations.*

![Factual Lookup Results](assets/Ques1.png)
![Factual Lookup Results](assets/Ques2.png)

---

### 2. Guardrail Enforcement (Questions 3, 4 & 5)
*Successfully triggered strict `"Not found"` guardrails for out-of-domain and subjective queries.*

![Guardrail Results](assets/Ques3.png)
![Guardrail Results](assets/Ques4.png)
![Guardrail Results](assets/Ques5.png)

---

## 🛠️ How to Run Locally

### 1. Prerequisites & Environment Setup
Clone the repository and install the required dependencies:
```bash
pip install fastapi uvicorn langchain langchain-chroma langchain-huggingface pydantic python-dotenv# GenAI-Project_4-Orient_BlackSwan_History_Bot
