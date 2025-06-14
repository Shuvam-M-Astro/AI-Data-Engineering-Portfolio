from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self):
        # Initialize the language model
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo"
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        
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
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
        # Create QA chain
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        Answer:"""
        
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
        
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }

def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Example usage
    print("RAG System initialized.")
    print("Loading example document...")
    
    # Load and process documents
    try:
        documents = rag.load_document("example.pdf")  # Replace with your document
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