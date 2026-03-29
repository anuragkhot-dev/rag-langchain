from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_chroma import Chroma
from graph import LangGraph
from pypdf import PdfReader
from io import BytesIO
import shutil
import uuid
import os

graph = LangGraph()
class LangChain :

    def extract_text_from_file(file):
        content = file.file.read()

        if file.filename.endswith(".pdf"):
            pdf = PdfReader(BytesIO(content))
            text = ""
            for page in pdf.pages :
                text += page.extract_text() or ""
        else :
                text = content.decode("utf-8")
        
        return text

    
    @staticmethod
    def ingest_document (file, query) :

        # Extract text from file
        text = LangChain.extract_text_from_file(file)
    
        # Split the text and form chunks
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=20
            )
        chunks = splitter.split_text(text)
        docs = [Document(page_content=c) for c in chunks]

        # Embeddings
        embedding_model = OllamaEmbeddings(model="llama3:8b")
        collection_name = f"rag_{uuid.uuid4()}"

        vector_store = Chroma(
            collection_name = collection_name,
            embedding_function = embedding_model,
            persist_directory = "./chroma_db"
        )

        vector_store.add_documents(docs)

        answer = graph.run(query, vector_store)

        return answer
        

