import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def main():
    # Define file and directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_filename = "History_2024-25.pdf"
    pdf_path = os.path.join(current_dir, pdf_filename)
    persist_directory = os.path.join(current_dir, "history_db")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    # 1. Load the PDF file using LangChain's PyPDFLoader
    print(f"Loading '{pdf_filename}' using PyPDFLoader...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from '{pdf_filename}'.")

    # 2. Split the text into chunks of 500 characters with 50 characters overlap
    print("Splitting text using RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    total_chunks_created = len(chunks)
    print(f"Total number of chunks created: {total_chunks_created}")

    # 3. Compute vector embeddings using HuggingFaceEmbeddings
    print("Computing vector embeddings using 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. Store embeddings in a local Chroma vector database folder 'history_db'
    print(f"Storing embeddings into Chroma vector database at '{persist_directory}'...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # 5. Confirming total chunks created and saved
    total_chunks_saved = vector_db._collection.count()
    print("\n--- Summary ---")
    print(f"Total chunks created: {total_chunks_created}")
    print(f"Total chunks saved to 'history_db': {total_chunks_saved}")
    print("Vector database ingestion completed successfully!")


if __name__ == "__main__":
    main()
