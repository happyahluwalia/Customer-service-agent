"""
Customer Service Agent with RAG Capabilities
This application provides a Streamlit interface for document management and question answering
using RAG (Retrieval Augmented Generation) with routing capabilities.
"""

import os
from typing import List, Dict, Any, Literal, Optional, Tuple
from dataclasses import dataclass
import streamlit as st
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import tempfile
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from langchain.schema import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing_extensions import TypeAlias

# Type definitions
DatabaseType: TypeAlias = Literal["products", "support", "finance"]
PERSIST_DIRECTORY = "db_storage"
VECTOR_SIZE = 1536  # OpenAI embedding size
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
SIMILARITY_K = 3
CONFIDENCE_THRESHOLD = 0.5

@dataclass
class CollectionConfig:
    """Configuration for each database collection"""
    name: str
    description: str
    collection_name: str

# Collection configurations
COLLECTIONS: Dict[DatabaseType, CollectionConfig] = {
    "products": CollectionConfig(
        name="Product Information",
        description="Product details, specifications, and features",
        collection_name="products_collection"
    ),
    "support": CollectionConfig(
        name="Customer Support & FAQ",
        description="Customer support information, frequently asked questions, and guides",
        collection_name="support_collection"
    ),
    "finance": CollectionConfig(
        name="Financial Information",
        description="Financial data, revenue, costs, and liabilities",
        collection_name="finance_collection"
    )
}

def init_session_state() -> None:
    """Initialize Streamlit session state variables"""
    session_vars = {
        'openai_api_key': "",
        'qdrant_url': "",
        'qdrant_api_key': "",
        'embeddings': None,
        'llm': None,
        'databases': {},
        'chat_history': []  # Added for maintaining conversation context
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def initialize_models() -> bool:
    """
    Initialize OpenAI models and Qdrant client
    Returns:
        bool: True if initialization successful, False otherwise
    """
    if not all([
        st.session_state.openai_api_key,
        st.session_state.qdrant_url,
        st.session_state.qdrant_api_key
    ]):
        return False
    
    try:
        # Set OpenAI API key and initialize models
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        st.session_state.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=st.session_state.openai_api_key
        )
        st.session_state.llm = ChatOpenAI(
            temperature=0,
            api_key=st.session_state.openai_api_key
        )
        
        # Initialize Qdrant client
        client = QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key
        )
        
        # Test connection and initialize collections
        client.get_collections()
        st.session_state.databases = {}
        
        for db_type, config in COLLECTIONS.items():
            try:
                # Get or create collection
                try:
                    client.get_collection(config.collection_name)
                except Exception:
                    client.create_collection(
                        collection_name=config.collection_name,
                        vectors_config=VectorParams(
                            size=VECTOR_SIZE,
                            distance=Distance.COSINE
                        )
                    )
                
                # Initialize Qdrant wrapper
                st.session_state.databases[db_type] = Qdrant(
                    client=client,
                    collection_name=config.collection_name,
                    embeddings=st.session_state.embeddings
                )
            except Exception as e:
                st.error(f"Failed to initialize {config.name}: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize models: {str(e)}")
        return False

def process_document(file) -> List[Document]:
    """
    Process uploaded PDF document
    Args:
        file: Uploaded PDF file
    Returns:
        List[Document]: List of processed document chunks
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and process document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return text_splitter.split_documents(documents)
        
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []

def create_routing_agent() -> Agent:
    """
    Creates a routing agent using phidata framework
    Returns:
        Agent: Configured routing agent
    """
    return Agent(
        model=OpenAIChat(
            id="gpt-4",  # Updated to use latest model
            api_key=st.session_state.openai_api_key
        ),
        tools=[],
        description="""You are a query routing expert. Your only job is to analyze questions and determine which database they should be routed to.
        You must respond with exactly one of these three options: 'products', 'support', or 'finance'. The user's question is: {question}""",
        instructions=[
            "Follow these rules strictly:",
            "1. For questions about products, features, specifications, or item details, or product manuals → return 'products'",
            "2. For questions about help, guidance, troubleshooting, or customer service, FAQ, or guides → return 'support'",
            "3. For questions about costs, revenue, pricing, or financial data, or financial reports and investments → return 'finance'",
            "4. Return ONLY the database name, no other text or explanation",
            "5. If you're not confident about the routing, return an empty response"
        ],
        markdown=False,
        show_tool_calls=False
    )

def route_query(question: str) -> Optional[DatabaseType]:
    """
    Route query to appropriate database using vector similarity and LLM fallback
    Args:
        question: User question
    Returns:
        Optional[DatabaseType]: Selected database type or None if no suitable match
    """
    try:
        # Vector similarity search
        best_score = -1
        best_db_type = None
        all_scores = {}
        
        for db_type, db in st.session_state.databases.items():
            results = db.similarity_search_with_score(
                question,
                k=SIMILARITY_K
            )
            
            if results:
                avg_score = sum(score for _, score in results) / len(results)
                all_scores[db_type] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_db_type = db_type
        
        # Check if vector similarity found a good match
        if best_score >= CONFIDENCE_THRESHOLD and best_db_type:
            st.success(f"Using vector similarity routing: {best_db_type} (confidence: {best_score:.3f})")
            return best_db_type
        
        # Fallback to LLM routing
        st.warning(f"Low confidence scores (below {CONFIDENCE_THRESHOLD}), falling back to LLM routing")
        routing_agent = create_routing_agent()
        response = routing_agent.run(question)
        
        db_type = (response.content
                  .strip()
                  .lower()
                  .translate(str.maketrans('', '', '`\'"')))
        
        if db_type in COLLECTIONS:
            st.success(f"Using LLM routing decision: {db_type}")
            return db_type
        
        st.warning("No suitable database found, will use web search fallback")
        return None
        
    except Exception as e:
        st.error(f"Routing error: {str(e)}")
        return None

def create_fallback_agent(chat_model: BaseLanguageModel):
    """
    Create a LangGraph agent for web research
    Args:
        chat_model: Language model for the agent
    Returns:
        Agent: Configured research agent
    """
    def web_research(query: str) -> str:
        """Web search with result formatting"""
        try:
            search = DuckDuckGoSearchRun(num_results=5)
            return search.run(query)
        except Exception as e:
            return f"Search failed: {str(e)}. Providing answer based on general knowledge."

    return create_react_agent(
        model=chat_model,
        tools=[web_research],
        debug=False
    )

def query_database(db: Qdrant, question: str) -> Tuple[str, List[Document]]:
    """
    Query the database and return answer with relevant documents
    Args:
        db: Qdrant database instance
        question: User question
    Returns:
        Tuple[str, List[Document]]: Answer and relevant documents
    """
    try:
        # Setup retriever
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        )

        relevant_docs = retriever.get_relevant_documents(question)

        if not relevant_docs:
            raise ValueError("No relevant documents found in database")

        # Create QA chain with improved prompt
        retrieval_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on provided context.
                         Always be direct and concise in your responses.
                         If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
                         Base your answers strictly on the provided context and avoid making assumptions.
                         Use bullet points for lists and structure your response for clarity."""),
            ("human", "Context:\n{context}"),
            ("human", "Question: {input}"),
            ("assistant", "I'll help answer your question based on the context provided."),
            ("human", "Please provide your answer:"),
        ])
        
        # Create and execute chain
        combine_docs_chain = create_stuff_documents_chain(
            st.session_state.llm,
            retrieval_qa_prompt
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        response = retrieval_chain.invoke({"input": question})
        return response['answer'], relevant_docs

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I encountered an error. Please try rephrasing your question.", []

def _handle_web_fallback(question: str) -> Tuple[str, List]:
    """
    Handle web search fallback when no relevant documents found
    Args:
        question: User question
    Returns:
        Tuple[str, List]: Answer and empty list (no documents)
    """
    st.info("No relevant documents found. Searching web...")
    fallback_agent = create_fallback_agent(st.session_state.llm)
    
    with st.spinner('Researching...'):
        agent_input = {
            "messages": [
                HumanMessage(content=f"Research and provide a detailed answer for: '{question}'")
            ],
            "is_last_step": False
        }
        
        try:
            response = fallback_agent.invoke(
                agent_input,
                config={"recursion_limit": 100}
            )
            if isinstance(response, dict) and "messages" in response:
                answer = response["messages"][-1].content
                return f"Web Search Result:\n{answer}", []
                
        except Exception as e:
            st.warning(f"Web search failed: {str(e)}. Falling back to general response.")
            fallback_response = st.session_state.llm.invoke(question).content
            return f"Web search unavailable. General response: {fallback_response}", []

def main():
    """Main application function"""
    # Setup page configuration
    st.set_page_config(
        page_title="Customer Service Agent with Database Routing",
        page_icon="📚",
        layout="wide"
    )
    st.title("📠 Customer Service Agent for Product, Financial or Support Enquiries")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API key inputs
        api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            value=st.session_state.openai_api_key,
            key="api_key_input"
        )
        
        qdrant_url = st.text_input(
            "Qdrant URL:",
            value=st.session_state.qdrant_url,
            help="Example: https://your-cluster.qdrant.tech"
        )
        
        qdrant_api_key = st.text_input(
            "Qdrant API Key:",
            type="password",
            value=st.session_state.qdrant_api_key
        )
        
        # Update session state
        if api_key:
            st.session_state.openai_api_key = api_key
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url
        if qdrant_api_key:
            st.session_state.qdrant_api_key = qdrant_api_key
        
        # Initialize models
        if all([
            st.session_state.openai_api_key,
            st.session_state.qdrant_url,
            st.session_state.qdrant_api_key
        ]):
            if initialize_models():
                st.success("Connected to OpenAI and Qdrant successfully!")
            else:
                st.error("Failed to initialize. Please check your credentials.")
                st.stop()
        else:
            st.warning("Please enter all required credentials to continue")
            st.stop()

        st.markdown("---")

   # Document upload section
    st.header("Document Upload")
    st.info("Upload documents to populate the databases. Each tab corresponds to a different database.")
    
    # Create tabs for different collections
    tabs = st.tabs([config.name for config in COLLECTIONS.values()])
    
    for (collection_type, config), tab in zip(COLLECTIONS.items(), tabs):
        with tab:
            st.write(config.description)
            
            # File uploader for each collection
            uploaded_files = st.file_uploader(
                f"Upload PDF documents to {config.name}",
                type="pdf",
                key=f"upload_{collection_type}",
                accept_multiple_files=True
            )
            
            if uploaded_files:
                with st.spinner('Processing documents...'):
                    # Process all uploaded files
                    all_texts = []
                    for uploaded_file in uploaded_files:
                        texts = process_document(uploaded_file)
                        if texts:  # Only add if processing was successful
                            all_texts.extend(texts)
                            st.success(f"Processed {uploaded_file.name}")
                        else:
                            st.error(f"Failed to process {uploaded_file.name}")
                    
                    # Add processed documents to database
                    if all_texts:
                        try:
                            db = st.session_state.databases[collection_type]
                            db.add_documents(all_texts)
                            st.success(f"Added {len(all_texts)} document chunks to {config.name}!")
                        except Exception as e:
                            st.error(f"Failed to add documents to database: {str(e)}")
    
    # Query section
    st.header("Ask Questions")
    st.info("""Enter your question below to find answers from the relevant database.
            The system will automatically route your question to the most appropriate knowledge base.""")
    
    # Question input with larger text area
    question = st.text_area(
        "Enter your question:",
        height=100,
        help="Type your question here. The system will automatically route it to the most relevant database."
    )
    
    # Add a submit button for better control
    if st.button("Submit Question", disabled=not question):
        with st.spinner('Finding answer...'):
            # Store question in chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Route the question
            collection_type = route_query(question)
            
            # Create containers for displaying results
            answer_container = st.container()
            context_container = st.container()
            
            with answer_container:
                if collection_type is None:
                    # Web search fallback
                    answer, _ = _handle_web_fallback(question)
                    st.write("### Answer (from web search)")
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    # Database query
                    st.info(f"Routing question to: {COLLECTIONS[collection_type].name}")
                    db = st.session_state.databases[collection_type]
                    answer, relevant_docs = query_database(db, question)
                    
                    st.write("### Answer")
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Display relevant documents in expandable section
                    with context_container:
                        with st.expander("View Source Documents"):
                            for i, doc in enumerate(relevant_docs, 1):
                                st.markdown(f"**Source {i}:**")
                                st.write(doc.page_content)
                                st.markdown("---")
    
    # Display chat history in an expandable section
    with st.expander("View Conversation History"):
        for message in st.session_state.chat_history:
            role_emoji = "🙋" if message["role"] == "user" else "🤖"
            st.markdown(f"{role_emoji} **{message['role'].title()}:** {message['content']}")
            st.markdown("---")

if __name__ == "__main__":
    main()