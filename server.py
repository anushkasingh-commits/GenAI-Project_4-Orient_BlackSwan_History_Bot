import os
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (.env)
load_dotenv()

# System prompt required for History Catalogue Assistant
SYSTEM_PROMPT = (
    "You are a history catalogue assistant. "
    "Answer the question using ONLY the provided context. "
    "Always include the source page reference or document section where the answer is found. "
    "If the answer is not present in the context, reply strictly with \"Not found\"."
)

# Global variables for loaded pipeline
rag_pipeline = {}


def get_chat_model():
    """Initializes and returns an LCEL-compatible chat model based on available environment keys."""
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Using Google Gemini model (gemini-3.5-flash-lite)...")
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
            google_api_key=google_api_key,
        )
    elif groq_api_key:
        from langchain_groq import ChatGroq
        print("Using Groq model (llama-3.3-70b-versatile)...")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=groq_api_key,
        )
    elif openai_api_key:
        from langchain_openai import ChatOpenAI
        print("Using OpenAI model (gpt-4o-mini)...")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key,
        )
    else:
        raise ValueError("No API key configured! Please set GOOGLE_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in .env.")


def format_docs(docs) -> str:
    """Formats retrieved document chunks with page references and deduplicates them."""
    formatted = []
    seen = set()
    for doc in docs:
        content = doc.page_content.strip()
        if content in seen:
            continue
        seen.add(content)
        page = doc.metadata.get("page", None)
        page_info = f"Page {page + 1}" if isinstance(page, int) else "Page Reference: Unknown"
        formatted.append(f"[{page_info}]\n{content}")
    return "\n\n".join(formatted)


def initialize_rag():
    """Loads history_db vector store, retriever, and builds the LCEL generation chain."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(current_dir, "history_db")

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Chroma vector store not found at '{persist_directory}'. Please run ingest.py first!"
        )

    print(f"Loading Chroma vector database from '{persist_directory}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    # Retriever fetching top 4 most relevant chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    # LLM Chat Model
    llm = get_chat_model()

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])

    generation_chain = prompt | llm | StrOutputParser()

    return {
        "vector_db": vector_db,
        "retriever": retriever,
        "generation_chain": generation_chain,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load RAG components once
    print("Starting up FastAPI History RAG server...")
    pipeline = initialize_rag()
    rag_pipeline.update(pipeline)
    print("History RAG pipeline loaded successfully and ready to serve requests.")
    yield
    # Shutdown
    rag_pipeline.clear()
    print("History RAG server shutdown.")


# Initialize FastAPI App
app = FastAPI(
    title="History Catalogue RAG API",
    description="FastAPI service for querying the History 2024-25 catalogue using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS middleware so local frontend web apps can call the API without errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g. http://localhost:3000, http://127.0.0.1:5500)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Pydantic Request & Response Models
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        description="The question to ask about the History catalogue",
        examples=["Who is the author of 'A Critical History of Women’s Health in Modern Sri Lanka'?"],
    )


class ChunkInfo(BaseModel):
    page: Optional[int] = Field(None, description="1-indexed source page number from the PDF")
    content: str = Field(..., description="Text content of the retrieved chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the chunk")


class AskResponse(BaseModel):
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer with source page reference or 'Not found'")
    retrieved_chunks: List[ChunkInfo] = Field(
        ..., description="List of top 4 retrieved chunks providing context"
    )


from fastapi.responses import FileResponse


@app.get("/")
def serve_index():
    """Serves the chatbot web interface."""
    index_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "status": "online",
        "service": "History Catalogue RAG API",
        "endpoints": {
            "ask": "POST /ask",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify server status."""
    return {
        "status": "online",
        "service": "History Catalogue RAG API",
        "endpoints": {
            "ask": "POST /ask",
            "docs": "GET /docs",
        },
    }


@app.post("/ask", response_model=AskResponse)
def ask_question_endpoint(req: QuestionRequest):
    """
    Accepts a question, retrieves the top 4 relevant chunks from history_db,
    runs the prompt through the LCEL RAG chain, and returns the answer and retrieved chunks.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retriever = rag_pipeline.get("retriever")
    generation_chain = rag_pipeline.get("generation_chain")

    if not retriever or not generation_chain:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized or still starting up.",
        )

    try:
        # 1. Retrieve top 4 most relevant chunks
        docs = retriever.invoke(req.question)

        # 2. Format context for prompt
        context_str = format_docs(docs)

        # 3. Generate answer via LCEL chain
        answer = generation_chain.invoke({
            "context": context_str,
            "question": req.question,
        })

        # 4. Format retrieved chunks for the response
        chunks = []
        for doc in docs:
            raw_page = doc.metadata.get("page")
            page_num = (raw_page + 1) if isinstance(raw_page, int) else None
            chunks.append(
                ChunkInfo(
                    page=page_num,
                    content=doc.page_content.strip(),
                    metadata=doc.metadata,
                )
            )

        return AskResponse(
            question=req.question,
            answer=answer,
            retrieved_chunks=chunks,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Run the server locally on http://127.0.0.1:8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
