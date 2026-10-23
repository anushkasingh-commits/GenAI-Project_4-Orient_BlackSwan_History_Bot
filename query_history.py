import os
import sys
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 5 Catalogue test questions (including positive catalogue lookups and a negative out-of-domain test)
CATALOGUE_QUESTIONS = [
    "Who is the author of 'A Critical History of Women’s Health in Modern Sri Lanka' and what department do they belong to?",
    "What is the book 'Discovering India Anew' about and who wrote it?",
    "Which publishing houses or presses is Orient BlackSwan the exclusive distributor for?",
    "What series is the book 'A Critical History of Women’s Health in Modern Sri Lanka' published under?",
    "What is the capital of France?",  # Negative test: should reply strictly with 'Not found'
]


def get_chat_model():
    """Initializes and returns an LCEL-compatible chat model based on available environment keys."""
    load_dotenv()

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
        # Prompt user if no API key is set in environment or .env
        print("\nNo API key found in environment variables (GOOGLE_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY).")
        try:
            choice = input("Select provider to provide API key (1: Google Gemini [default], 2: Groq): ").strip()
        except EOFError:
            choice = "1"

        if choice == "2":
            key = input("Enter your GROQ_API_KEY: ").strip()
            os.environ["GROQ_API_KEY"] = key
            from langchain_groq import ChatGroq
            return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=key)
        else:
            key = input("Enter your GOOGLE_API_KEY / GEMINI_API_KEY: ").strip()
            os.environ["GOOGLE_API_KEY"] = key
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, google_api_key=key)


def format_docs(docs):
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


def build_rag_chain():
    """Loads history_db, creates a top-4 retriever, and builds the LCEL RAG chain."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(current_dir, "history_db")

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Chroma database folder '{persist_directory}' does not exist. Please run ingest.py first!"
        )

    # 1. Load Chroma vector database with embeddings
    print(f"Loading Chroma database from '{persist_directory}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    # 2. Create retriever fetching the top 4 most relevant chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    # 3. Chat model
    llm = get_chat_model()

    # 4. System prompt instruction
    system_prompt = (
        "You are a history catalogue assistant. "
        "Answer the question using ONLY the provided context. "
        "Always include the source page reference or document section where the answer is found. "
        "If the answer is not present in the context, reply strictly with \"Not found\"."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])

    # 5. Build LangChain LCEL RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def ask_question(rag_chain, question: str):
    """Executes a question through the RAG chain and prints formatted output."""
    print(f"\nQ: {question}")
    response = rag_chain.invoke(question)
    print(f"A: {response}")
    print("-" * 60)
    return response


def run_test_suite(rag_chain):
    """Runs the 5 predefined catalogue test questions."""
    print("\n" + "=" * 60)
    print("RUNNING 5 CATALOGUE TEST QUESTIONS")
    print("=" * 60)
    for idx, q in enumerate(CATALOGUE_QUESTIONS, start=1):
        print(f"\n[Question {idx}/5]")
        ask_question(rag_chain, q)


def main():
    rag_chain, retriever = build_rag_chain()

    # Check command-line flags for automatic testing
    if "--auto" in sys.argv or "-a" in sys.argv or not sys.stdin.isatty():
        run_test_suite(rag_chain)
        return

    # Interactive Loop
    while True:
        print("\n" + "=" * 40)
        print("      HISTORY CATALOGUE ASSISTANT")
        print("=" * 40)
        print("1. Run the 5 Catalogue Test Questions automatically")
        print("2. Enter a custom question in terminal")
        print("3. Exit")
        try:
            choice = input("\nEnter choice (1, 2, or 3) [default: 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if choice in ("", "1"):
            run_test_suite(rag_chain)
        elif choice == "2":
            print("\nType your question (or 'back' to return to menu, 'exit' to quit):")
            while True:
                try:
                    user_q = input("\nAsk question > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    return
                if user_q.lower() == "back":
                    break
                ask_question(rag_chain, user_q)
        elif choice == "3" or choice.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
