from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import logging

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        # Initialize the language model
        self.llm = ChatOpenAI(
            temperature=0.0,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize vector store
        self.vector_store = None
        
        # Initialize QA chain
        self.qa_chain = None
    
    def load_document(self, file_path):
        """Load document from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        documents = loader.load()
        return documents
    
    def process_documents(self, documents):
        """Process documents and create vector store"""
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        # Guard against empty docs
        if not texts:
            raise ValueError("No text chunks produced from documents. Check the loader or text splitter settings.")
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
        # Create QA chain
        prompt_template = (
            "You are a helpful assistant. Use the context to answer the question. "
            "If unsure, say you don't know.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        )
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def query(self, question):
        """Query the RAG system"""
        if not self.qa_chain:
            return "Please load and process documents first."

        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result.get("source_documents", [])]
            }
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return {"answer": "An error occurred while processing the query.", "sources": []}

def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Example usage
    print("RAG System initialized.")
    print("Loading example document...")
    
    # Load and process documents
    try:
        # Replace with your document path or set DOC_PATH env var
        doc_path = os.getenv("DOC_PATH", "example.pdf")
        documents = rag.load_document(doc_path)
        rag.process_documents(documents)
        print("Documents processed successfully.")
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        return
    
    # Interactive query loop
    print("\nType 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
        
        result = rag.query(question)
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source[:200]}...")

if __name__ == "__main__":
    main() 