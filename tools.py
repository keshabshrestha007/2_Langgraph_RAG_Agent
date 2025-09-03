import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec,Pinecone

load_dotenv()

if os.getenv("PINECONE_API_KEY") is None:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

pinecone_api_key = os.getenv("PINECONE_API_KEY")

def retriever_tool(pdf_dir):
    index_name = "project-1-multistep-rag"

    pc = Pinecone(api_key= pinecone_api_key)
    
    # Ensure index exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

       
    # Wait until index is ready
    while True:
        status = pc.describe_index(index_name).status
        if status.get("ready", False):
            break
        time.sleep(2)
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2"))

    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    # Check if vector index is empty
    is_empty = sum(stats["total_vector_count"] for _ in stats["namespaces"].values()) == 0

    if is_empty:
        documents = []
        for pdf_file in pdf_dir:
            try:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                for doc in docs:
                    if doc.page_content.strip():
                        doc.metadata["source"] = pdf_file
                        documents.append(doc)
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2"),
            index_name=index_name
        ) 
   

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
      