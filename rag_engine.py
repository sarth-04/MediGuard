from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
import config
import data_loader
import os
import shutil

class RAGPipeline:
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL, 
            temperature=0, 
            google_api_key=api_key
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL, 
            google_api_key=api_key
        )
        self.vectorstore = None
        self.retriever = None
        
        # Initial load
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize existing store or create default one."""
        self.vectorstore = Chroma(
            persist_directory=config.VECTOR_STORE_PATH,
            embedding_function=self.embeddings,
            collection_name="health_knowledge"
        )
        
        if self.vectorstore._collection.count() == 0:
            print("DB Empty. Loading default data...")
            self._process_and_add_docs(data_loader.get_documents(None))
            
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

    def rebuild_index(self, new_file_path):
        """Forces a complete rebuild of the vector database."""
        print(f"🔄 Rebuilding index with: {new_file_path}")
        try:
            self.vectorstore.delete_collection()
        except Exception as e:
            print(f"Warning clearing collection: {e}")

        self.vectorstore = Chroma(
            persist_directory=config.VECTOR_STORE_PATH,
            embedding_function=self.embeddings,
            collection_name="health_knowledge"
        )

        docs = data_loader.get_documents(new_file_path)
        self._process_and_add_docs(docs)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        print("✅ Index rebuild complete.")

    def _process_and_add_docs(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)

    def get_conversational_chain(self):
        """
        Creates a chain using Pure LCEL (No langchain.chains dependency).
        """
        # --- 1. Contextualize Question Logic ---
        # If chat history exists, rewrite the question. If not, use the input as is.
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Chain to rewrite query: Prompt -> LLM -> Text
        history_aware_rewriter = contextualize_q_prompt | self.llm | StrOutputParser()

        # Branching Logic:
        def get_chat_history(x):
            return x.get("chat_history", [])

        # Creates the "History Aware Retriever" manually
        retriever_chain = RunnableBranch(
            (
                lambda x: len(get_chat_history(x)) > 0, 
                history_aware_rewriter | self.retriever
            ),
            (lambda x: x["input"]) | self.retriever
        )

        # --- 2. Answer Logic (The "Stuff" Chain) ---
        qa_system_prompt = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Helper to format the list of docs into a string
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        # The Answer Chain: Formats context -> Prompts LLM -> Parses Output
        qa_chain = (
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["context"]))
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        # --- 3. Final RAG Pipeline ---
        # FIXED: We use .assign() instead of | 
        # This ensures we return a DICTIONARY {context: [...], answer: "..."}
        rag_chain = (
            RunnablePassthrough.assign(
                context=retriever_chain
            )
            .assign(answer=qa_chain)
        )

        return rag_chain