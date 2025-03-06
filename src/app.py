import os
from typing import List
import dotenv
import re
import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from chainlit.input_widget import Select, Slider
from pinecone import Pinecone
from langchain_community.vectorstores import FAISS
import tempfile

dotenv.load_dotenv()

# Initialize Pinecone with the new pattern
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

pdf_data = {
    "WHO guideline on control and elimination of human schistosomiasis (2022)": "https://iris.who.int/bitstream/handle/10665/351856/9789240041608-eng.pdf",
    "WHO guidelines for malaria (2023)": "https://iris.who.int/bitstream/handle/10665/373339/WHO-UCN-GMP-2023.01-Rev.1-eng.pdf"
}

@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Choose Model",
                values=[
                    # OpenAI models
                    "gpt-4o-mini",  # Default OpenAI model
                    "gpt-4o",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                    # Groq models
                    "llama-3.1-8b-instant", 
                    "llama-3.1-70b-versatile", 
                    "llama-3.3-70b-versatile",
                    'llama3-70b-8192', 
                    'llama3-8b-8192',
                    'mixtral-8x7b-32768',
                    'gemma2-9b-it'
                ],
                initial_index=0,  # Set gpt-4o-mini as default
            ),
            Slider(
                id="temperature",
                label="Model - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            )
        ]
    ).send()
    
    await setup_agent(settings)
    
    # Let the user know that the system is ready
    await cl.Message(content="Hello, I am here to help you with questions on your provided PDF files.").send()


@cl.on_settings_update
async def setup_agent(settings):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    try:
        # Try to use Pinecone with the new client
        index_name = os.getenv("PINECONE_INDEX_NAME")
        namespace = os.getenv("PINECONE_NAME_SPACE", "")
        
        # Get the index from the Pinecone client
        index = pc.Index(index_name)
        
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=namespace
        )
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        print("Falling back to local vector store...")
        
        # Create a simple local vector store with sample data
        texts = [
            "WHO guidelines recommend preventive chemotherapy as the main strategy for schistosomiasis control.",
            "Malaria prevention includes the use of insecticide-treated mosquito nets and indoor residual spraying.",
            # Add more sample texts if needed
        ]
        
        vector_store = FAISS.from_texts(texts, embeddings)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_type="similarity")
    
    # Initialize LLM based on settings
    if settings['model'] == "gpt-3.5-turbo" or settings['model'].startswith("gpt-4"):
        # Use the specified OpenAI model or default to gpt-4o-mini
        openai_model = settings['model'] if settings['model'].startswith("gpt") else "gpt-4o-mini"
        llm = ChatOpenAI(
            model=openai_model,
            temperature=settings['temperature'],
            streaming=True
        )
    else:
        llm = ChatGroq(
            model=settings['model'],
            temperature=settings['temperature'],
            streaming=True,
            api_key=os.getenv("GROQ_API_KEY")
        )
    
    # Create a contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("human", "Chat History: {chat_history}\nUser question: {input}")
        ]
    )
    
    # Create history-aware retriever without the input_key parameter
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        contextualize_q_prompt
    )
    
    # Create QA prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise.\n\n"
        "Context: {context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}")
        ]
    )
    
    # Create document chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )
    
    # Store the chain and chat history in the user session
    cl.user_session.set("chain", rag_chain)
    cl.user_session.set("chat_history", [])


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    chat_history = cl.user_session.get("chat_history")
    
    cb = cl.AsyncLangchainCallbackHandler()
    
    # Call the chain with the input key that the history_aware_retriever expects
    response = await chain.ainvoke(
        {
            "input": message.content,  # Use "input" to match what history_aware_retriever expects
            "chat_history": chat_history
        },
        callbacks=[cb]
    )
    
    answer = response["answer"]
    source_documents = response.get("context", [])  # Get source documents from context
    
    if source_documents:
        # Create a list of unique source citations with links
        source_citations = []
        for doc in source_documents:
            if doc.metadata.get('page') and doc.metadata.get('source'):
                pdf_title = doc.metadata['source']
                # Extract the PDF title and year using regex
                match = re.search(r"/(.+?)\s*(\(\d{4}\))\.pdf$", pdf_title)
                if match:
                    pdf_title_short = match.group(1)
                    year = match.group(2)
                    page_num = int(doc.metadata['page'])
                    citation = f"{pdf_title_short} {year}, Page {page_num}"
                    
                    # Get the URL from the pdf_data dictionary
                    url = pdf_data.get(f"{pdf_title_short} {year}", "")
                    if url:
                        # Create a clickable link to the exact page
                        citation_link = f"[{citation}]({url}#page={page_num})"
                        source_citations.append(citation_link)
                    else:
                        source_citations.append(citation)
        
        source_citations = list(set(source_citations))
        
        if source_citations:
            answer += "\n\nSources:\n" + "\n".join(source_citations)
        else:
            answer += "\nNo sources found"
    
    # Send the message without elements
    await cl.Message(content=answer).send()
    
    # Update chat history
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=answer))
    cl.user_session.set("chat_history", chat_history)