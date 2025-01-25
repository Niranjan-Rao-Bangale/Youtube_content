# pip install langchain_community langchain chromadb pypdf requests beautifulsoup4 google-generativeai chroma-migrate

import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

import langchain
from langchain.llms.base import LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Import Google's generative AI client
import google.generativeai as genai
from chromadb.config import Settings
#####################
# A) Gemini LLM Wrapper
#####################

class OllamaLLM(LLM):
    """
    Minimal Ollama wrapper for LangChain.
    We'll call 'ollama' CLI in a subprocess.
    The model must be installed locally (e.g. llama3.2).

    You might also have the 'ollama' Python library.
    For demonstration, we do CLI calls to 'ollama'.
    """

    model_name: str = "deepseek-r1"

    @property
    def _llm_type(self):
        return "ollama"

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        """
        Sends the prompt to the Ollama CLI.
        """
        # Build arguments
        cmd = [
            "ollama",
            "run",
            self.model_name,
        ]

        # If we have stop sequences, we might handle them manually.
        # For simplicity, ignoring that here.

        # We'll store the prompt in a temporary file or pass it directly
        # Ollama CLI supports: `ollama run deepseek-r1 --prompt "some text"`
        # but there's also streaming output. Let's just capture it.

        # Because `--prompt` is single-line, let's do a simpler approach:
        # or we can just do `echo prompt | ollama run`.
        # We'll do a direct approach here:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )

        output, error = process.communicate(input=prompt)
        print("DEBUG: Raw LLM Output:", output)
        print("DEBUG: Raw LLM Error:", error)
        if error:
            print("Ollama error:", error)

        return output.strip()

class GeminiLLM(LLM):
    """
    Minimal Google Generative AI wrapper for LangChain.
    We call the 'gemini-1.5-flash' model using google.generativeai.
    """

    model_name: str = "gemini-1.5-flash"   # default model
    api_key: str = "##########################" # replace ########################## with your api key

    @property
    def _llm_type(self) -> str:
        return "google_generativeai"

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        # Configure the generative AI client with your API key
        genai.configure(api_key=self.api_key)

        # Create a GenerativeModel object referencing the gemini-1.5-flash model
        model = genai.GenerativeModel(self.model_name)

        # Call the model
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text.strip()
        else:
            return ""

#######################
# B) Parse Documents
#######################

def parse_text_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def parse_pdf_file(filepath: str) -> str:
    """
    Extract text from PDF using pypdf.
    """
    pdf_reader = PdfReader(filepath)
    all_text = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    return "\n".join(all_text)

def parse_webpage(url: str) -> str:
    """
    Fetches a webpage, extracts visible text with BeautifulSoup.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.extract()
    text = soup.get_text(separator="\n")
    return text

######################
# C) Build Vector DB
######################



class DummyEmbeddings(Embeddings):
    """
    Simple dummy embeddings for demonstration.
    In real usage, swap out with local embeddings
    (sentence-transformers, Instructor, etc.).
    """
    def embed_documents(self, texts):
        return [[float(len(t))] * 10 for t in texts]

    def embed_query(self, text):
        return [float(len(text))] * 10

def ingest_documents(file_list=[], url_list=[]):
    """
    Convert each file/webpage into a list of chunked Documents.
    """
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Parse local files
    if file_list:
        for path in file_list:
            if path.lower().endswith(".pdf"):
                raw_text = parse_pdf_file(path)
            elif path.lower().endswith(".txt"):
                raw_text = parse_text_file(path)
            else:
                print(f"Skipping unsupported file type: {path}")
                continue

            chunks = text_splitter.split_text(raw_text)
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata={"source": path})
                docs.append(doc)

    # Parse web pages
    if url_list:
        for url in url_list:
            try:
                raw_text = parse_webpage(url)
                if raw_text.strip():
                    chunks = text_splitter.split_text(raw_text)
                    for chunk in chunks:
                        doc = Document(page_content=chunk, metadata={"source": url})
                        docs.append(doc)
            except Exception as e:
                print(f"Failed to process URL {url}: {e}")
                continue             
    return docs

##############################
# D) Main: Q&A with Gemini
##############################

def build_qa_system(google_api_key, file_list=None, url_list=None):
    """
    1. Parse docs
    2. Embed + store in Chroma
    3. Create RetrievalQA chain with our Gemini LLM
    """
    docs = ingest_documents(file_list, url_list)
    print("DEBUG: Number of docs:", len(docs))
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Provide client settings to specify tenant, directory, etc.
    # client_settings = Settings(
    #    chroma_db_impl="duckdb+parquet",
    #    persist_directory="./chroma_db",  # or your chosen directory
    #    anonymized_telemetry=False,
    #    tenant_id="default_tenant"        # specify tenant so it can connect
    #)
    # Create a Chroma vectorstore
    db = Chroma.from_documents(docs, 
    embedding=embedding_model, 
    collection_name="my_collection"
    # ,client_settings=client_settings
    )

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs_fetched = retriever.get_relevant_documents("test query")
    print("DEBUG: Fetched docs:", len(docs_fetched))

    # Our LLM (Gemini)
    gemini_llm = GeminiLLM(
        model_name="gemini-1.5-flash",
        api_key=google_api_key
    )

    # Build a standard "stuff" Q&A chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=gemini_llm,
        retriever=retriever,
        chain_type="refine"
    )

    return qa_chain
    

def build_qa_system_with_deepseek(file_list, url_list):
    """
    1. Parse docs
    2. Embed + store in Chroma
    3. Create RetrievalQA chain with our Ollama LLM
    """
    docs = ingest_documents(file_list, url_list)
    print("DEBUG: Number of docs:", len(docs))

    # Create a Chroma vectorstore
    db = Chroma.from_documents(docs, embedding=DummyEmbeddings(), collection_name="my_collection")

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs_fetched = retriever.get_relevant_documents("test query")
    print("DEBUG: Fetched docs:", len(docs_fetched))
    # Our LLM (Ollama)
    ollama_llm = OllamaLLM(
        model_name="deepseek-r1"
    )

    # Build a standard "StuffDocuments" Q&A chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama_llm,
        retriever=retriever,
        chain_type="stuff",  # or "map_reduce", "refine", etc.
    )

    return qa_chain

if __name__ == "__main__":
    # Example usage
    # Provide some local files and URLs
    my_files = []
    my_urls = ["https://www.paulgraham.com/greatwork.html"]

    # Replace with your real Google Generative AI API key
    

    # 1) Build system
    qa_system = build_qa_system(google_api_key="XXXXXXXXXXXXXXXXXXXXXXXXX", my_files=None, my_urls=None) # replase XXXXXXXXXXXXXXXXXXXXXXXXX with your Gemini api key
    print(f"qa_system: {qa_system}")

    # 2) Ask a question
    question = "Can you give me a summary of these documents?"
    # Using 'invoke' with a dict. The chain expects a 'query' or 'input' key.
    answer = qa_system.invoke({"query": question})

    print(f"Q: {question}\nA: {answer}")
