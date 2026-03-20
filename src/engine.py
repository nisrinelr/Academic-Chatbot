import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate


from utils import clean_text

# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# PDF Processing
# --------------------------------------------------
def process_pdf(pdf_path: str) -> list[str]:
    """Read PDF, clean text, and chunk it."""
    reader = PdfReader(pdf_path)
    raw_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + " "

    cleaned_text = clean_text(raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    return splitter.split_text(cleaned_text)


# --------------------------------------------------
# Embeddings
# --------------------------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# Vectorstore (cached on disk per PDF)
# --------------------------------------------------
def get_vectorstore(chunks: list[str], pdf_hash: str):
    embeddings = get_embeddings()
    vectorstore_dir = f"faiss_index_{pdf_hash}"

    if os.path.exists(vectorstore_dir):
        return FAISS.load_local(
            vectorstore_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(vectorstore_dir)
    return vectorstore


# --------------------------------------------------
# Chatbot / RAG Pipeline
# --------------------------------------------------
def initialize_chatbot(chunks: list[str], pdf_hash: str):
    vectorstore = get_vectorstore(chunks, pdf_hash)

    # Initialisation de Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )


    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5
    )

    # System prompt — respond in the same language as the question
    system_prompt = (
        "You are a helpful university academic advisor. "
        "Detect the language of the question and always answer in the same language. "
        "Supported languages: English, French, Arabic, and Moroccan Darija. "
        "Use only the context below to answer if the user wants to chat with you. Be polite and use a friendly tone. If the answer is not in the context, say so.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer (in the same language as the question):"
    )
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return qa_chain