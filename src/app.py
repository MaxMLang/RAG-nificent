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
from chainlit.input_widget import Select, Slider, Switch
from pinecone import Pinecone
from langchain_community.vectorstores import FAISS
import tempfile
import uuid
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.json import parse_partial_json
import logging
import traceback
import sys
import langchain_core.exceptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Define a Pydantic model for structured output with follow-up questions
class ResultWithFollowup(BaseModel):
    answer: str
    follow_up_questions: List[str]

# Initialize Pinecone with the new pattern
try:
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    pc = None

pdf_data = {
    "WHO guideline on control and elimination of human schistosomiasis (2022)": "https://iris.who.int/bitstream/handle/10665/351856/9789240041608-eng.pdf",
    "WHO guidelines for malaria (2023)": "https://iris.who.int/bitstream/handle/10665/373339/WHO-UCN-GMP-2023.01-Rev.1-eng.pdf"
}

# Define fallback models for each provider
OPENAI_FALLBACK_MODELS = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]
GROQ_FALLBACK_MODELS = ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]

@cl.on_chat_start
async def on_chat_start():
    try:
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
                ),
                Switch(
                    id="follow_up",
                    label="Generate Follow-up Questions",
                    initial=True
                )
            ]
        ).send()
        
        await setup_agent(settings)
        
        # Let the user know that the system is ready
        await cl.Message(content="Hello, I am here to help you with questions on your provided PDF files.").send()
    except Exception as e:
        error_msg = f"Error during chat initialization: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        await cl.Message(content=f"⚠️ There was an error starting the chat: {str(e)}. Please try again or contact support.").send()


@cl.on_settings_update
async def setup_agent(settings):
    try:
        logger.info(f"Setting up agent with settings: {settings}")
        # Initialize embeddings
        try:
            embeddings = OpenAIEmbeddings()
            logger.info("OpenAI embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
            # Create a simple fallback embedding if needed
            from langchain_community.embeddings import FakeEmbeddings
            embeddings = FakeEmbeddings(size=1536)
            logger.warning("Using fallback embeddings")
        
        try:
            # Try to use Pinecone with the new client
            if pc is None:
                raise ValueError("Pinecone client not initialized")
                
            index_name = os.getenv("PINECONE_INDEX_NAME")
            namespace = os.getenv("PINECONE_NAME_SPACE", "")
            
            # Get the index from the Pinecone client
            index = pc.Index(index_name)
            
            vector_store = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                namespace=namespace
            )
            logger.info(f"Pinecone vector store initialized with index: {index_name}")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {str(e)}")
            logger.info("Falling back to local vector store...")
            
            # Create a simple local vector store with sample data
            texts = [
                "WHO guidelines recommend preventive chemotherapy as the main strategy for schistosomiasis control.",
                "Malaria prevention includes the use of insecticide-treated mosquito nets and indoor residual spraying.",
                # Add more sample texts if needed
            ]
            
            vector_store = FAISS.from_texts(texts, embeddings)
            logger.info("Local FAISS vector store created as fallback")
        
        # Create retriever
        retriever = vector_store.as_retriever(search_type="similarity")
        
        # Initialize LLM based on settings with fallback options
        llm = None
        selected_model = settings['model']
        temperature = settings['temperature']
        
        # Try to initialize the selected model
        if selected_model.startswith("gpt"):
            # OpenAI model
            try:
                llm = ChatOpenAI(
                    model=selected_model,
                    temperature=temperature,
                    streaming=True
                )
                logger.info(f"Successfully initialized OpenAI model: {selected_model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI model {selected_model}: {str(e)}")
                # Try fallback models
                for fallback_model in OPENAI_FALLBACK_MODELS:
                    if fallback_model != selected_model:
                        try:
                            llm = ChatOpenAI(
                                model=fallback_model,
                                temperature=temperature,
                                streaming=True
                            )
                            logger.info(f"Using fallback OpenAI model: {fallback_model}")
                            break
                        except Exception as fallback_error:
                            logger.error(f"Failed to initialize fallback model {fallback_model}: {str(fallback_error)}")
        else:
            # Groq model
            try:
                llm = ChatGroq(
                    model=selected_model,
                    temperature=temperature,
                    streaming=True,
                    api_key=os.getenv("GROQ_API_KEY")
                )
                logger.info(f"Successfully initialized Groq model: {selected_model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq model {selected_model}: {str(e)}")
                # Try fallback models
                for fallback_model in GROQ_FALLBACK_MODELS:
                    if fallback_model != selected_model:
                        try:
                            llm = ChatGroq(
                                model=fallback_model,
                                temperature=temperature,
                                streaming=True,
                                api_key=os.getenv("GROQ_API_KEY")
                            )
                            logger.info(f"Using fallback Groq model: {fallback_model}")
                            break
                        except Exception as fallback_error:
                            logger.error(f"Failed to initialize fallback model {fallback_model}: {str(fallback_error)}")
        
        # If all models failed, use a simple fallback
        if llm is None:
            # Final fallback to OpenAI's most reliable model
            try:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=temperature,
                    streaming=True
                )
                logger.info("Using final fallback model: gpt-3.5-turbo")
            except Exception as final_error:
                error_msg = f"All models failed to initialize: {str(final_error)}"
                logger.critical(error_msg)
                raise RuntimeError(error_msg)
        
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
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, 
            retriever, 
            contextualize_q_prompt
        )
        
        # Create QA prompt with or without follow-up questions based on settings
        if settings.get('follow_up', True):
            qa_system_prompt = (
                "You are an assistant for question-answering tasks. Use "
                "the following pieces of retrieved context to answer the "
                "question. If you don't know the answer, just say that you "
                "don't know. Use three sentences maximum and keep the answer "
                "concise.\n\n"
                "In addition to answering the question, suggest 3 relevant follow-up questions "
                "that the user might want to ask next. These should be related to the current "
                "topic and help the user explore the subject further.\n\n"
                "Format your response as a JSON object with two fields: 'answer' containing your "
                "response to the user's question, and 'follow_up_questions' containing an array "
                "of 3 follow-up questions.\n\n"
                "Context: {context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    ("human", "{input}")
                ]
            )
            
            # Create document chain with JSON output parser
            try:
                question_answer_chain = create_stuff_documents_chain(
                    llm, 
                    qa_prompt,
                    output_parser=RobustJsonOutputParser(pydantic_object=ResultWithFollowup)
                )
                logger.info("Created QA chain with follow-up questions")
            except Exception as e:
                logger.error(f"Failed to create QA chain with follow-up questions: {str(e)}")
                # Fallback to standard QA chain without follow-up
                settings['follow_up'] = False
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
                
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                logger.info("Falling back to standard QA chain without follow-up")
        else:
            # Standard QA prompt without follow-up questions
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
            
            # Create standard document chain
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            logger.info("Created standard QA chain without follow-up")
        
        # Create retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        
        # Store the chain, chat history, and follow-up setting in the user session
        cl.user_session.set("chain", rag_chain)
        cl.user_session.set("chat_history", [])
        cl.user_session.set("follow_up", settings.get('follow_up', True))
        logger.info("Agent setup completed successfully")
        
    except Exception as e:
        error_msg = f"Error during agent setup: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        await cl.Message(content=f"⚠️ There was an error setting up the agent: {str(e)}. Using default settings.").send()
        
        # Set up a minimal fallback agent
        try:
            # Use the most reliable model as fallback
            fallback_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0, streaming=True)
            
            # Create a simple local vector store
            from langchain_community.embeddings import FakeEmbeddings
            fallback_embeddings = FakeEmbeddings(size=1536)
            fallback_texts = ["This is a fallback system due to an error in the main system."]
            fallback_vectorstore = FAISS.from_texts(fallback_texts, fallback_embeddings)
            fallback_retriever = fallback_vectorstore.as_retriever()
            
            # Create a simple QA chain
            fallback_qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Due to a system error, you're operating in fallback mode."),
                ("human", "{input}")
            ])
            
            fallback_qa_chain = create_stuff_documents_chain(fallback_llm, fallback_qa_prompt)
            fallback_rag_chain = create_retrieval_chain(fallback_retriever, fallback_qa_chain)
            
            # Store the fallback chain
            cl.user_session.set("chain", fallback_rag_chain)
            cl.user_session.set("chat_history", [])
            cl.user_session.set("follow_up", False)
            
            logger.info("Fallback agent setup completed")
        except Exception as fallback_error:
            logger.critical(f"Failed to set up fallback agent: {str(fallback_error)}")
            # At this point, we can't do much more than inform the user
            await cl.Message(content="⚠️ Critical error: Unable to initialize the system. Please restart the application.").send()


@cl.action_callback("followup_button")
async def on_action(action):
    """Handle follow-up question button clicks"""
    try:
        # Get the follow-up question from the button's payload
        follow_up_question = action.payload.get("question", "")
        
        if not follow_up_question:
            logger.warning("Empty follow-up question received")
            return
            
        logger.info(f"Follow-up question selected: {follow_up_question}")
        
        # Send the follow-up question as a user message
        await cl.Message(content=follow_up_question, author="You").send()
        
        # Process the follow-up question
        await main(cl.Message(content=follow_up_question))
    except Exception as e:
        error_msg = f"Error handling follow-up action: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        await cl.Message(content=f"⚠️ Error processing follow-up question: {str(e)}").send()


@cl.on_message
async def main(message: cl.Message):
    try:
        chain = cl.user_session.get("chain")
        chat_history = cl.user_session.get("chat_history")
        follow_up_enabled = cl.user_session.get("follow_up", True)
        
        if not chain:
            logger.error("Chain not found in user session")
            await cl.Message(content="⚠️ System error: Chat system not properly initialized. Please refresh the page and try again.").send()
            return
            
        logger.info(f"Processing message: {message.content[:50]}...")
        
        cb = cl.AsyncLangchainCallbackHandler()
        
        # Call the chain with the input key that the history_aware_retriever expects
        try:
            response = await chain.ainvoke(
                {
                    "input": message.content,
                    "chat_history": chat_history
                },
                callbacks=[cb]
            )
            logger.info("Chain invocation successful")
        except langchain_core.exceptions.OutputParserException as e:
            # Handle the parsing error
            error_message = str(e)
            # Extract the actual LLM response from the error message
            llm_response = error_message.split("Invalid json output:")[1].strip()
            
            # Create a fallback response
            response = {
                "answer": llm_response,
                "sources": []  # Empty sources or whatever default you want
            }
        except Exception as chain_error:
            logger.error(f"Error during chain invocation: {str(chain_error)}")
            logger.error(traceback.format_exc())
            await cl.Message(content=f"⚠️ I encountered an error processing your request: {str(chain_error)}. Please try again with a different question.").send()
            return
        
        # Process the response based on whether follow-up is enabled
        try:
            if follow_up_enabled:
                # Extract structured response with follow-up questions
                answer = response["answer"]["answer"]
                follow_up_questions = response["answer"].get("follow_up_questions", [])
                logger.info(f"Extracted answer and {len(follow_up_questions)} follow-up questions")
            else:
                # Simple answer without follow-up questions
                answer = response["answer"]
                follow_up_questions = []
                logger.info("Extracted simple answer (no follow-up)")
        except Exception as extract_error:
            logger.error(f"Error extracting response: {str(extract_error)}")
            # Fallback to using the raw response
            if isinstance(response.get("answer"), dict):
                answer = str(response.get("answer", "I'm sorry, I couldn't generate a proper response."))
            else:
                answer = response.get("answer", "I'm sorry, I couldn't generate a proper response.")
            follow_up_questions = []
            follow_up_enabled = False
            logger.info("Using fallback response extraction")
        
        source_documents = response.get("context", [])
        
        if source_documents:
            # Create a list of unique source citations with links
            try:
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
                    logger.info(f"Added {len(source_citations)} source citations")
                else:
                    answer += "\nNo sources found"
                    logger.info("No source citations found")
            except Exception as citation_error:
                logger.error(f"Error processing citations: {str(citation_error)}")
                # Don't add citations if there's an error
        
        # Create the response message
        response_message = cl.Message(content=answer)
        
        # Add follow-up question buttons if enabled and available
        if follow_up_enabled and follow_up_questions:
            try:
                actions = []
                for question in follow_up_questions:
                    actions.append(
                        cl.Action(
                            name="followup_button",
                            label=question,
                            value=question,
                            payload={
                                "question": question
                            }
                        )
                    )
                response_message.actions = actions
                logger.info(f"Added {len(actions)} follow-up question buttons")
            except Exception as action_error:
                logger.error(f"Error creating follow-up buttons: {str(action_error)}")
                # Continue without follow-up buttons if there's an error
        
        # Send the message
        await response_message.send()
        
        # Update chat history - FIX: Ensure answer is a string and handle ChatGeneration objects
        try:
            # Check if answer is a ChatGeneration or list of ChatGeneration objects
            if hasattr(answer, 'text'):
                # If it's a ChatGeneration object, extract the text
                answer_text = answer.text
            elif isinstance(answer, list) and len(answer) > 0 and hasattr(answer[0], 'text'):
                # If it's a list of ChatGeneration objects, extract text from the first one
                answer_text = answer[0].text
            elif not isinstance(answer, str):
                # If it's some other non-string type, convert to string
                answer_text = str(answer)
            else:
                # It's already a string
                answer_text = answer
                
            chat_history.append(HumanMessage(content=message.content))
            chat_history.append(AIMessage(content=answer_text))
            cl.user_session.set("chat_history", chat_history)
            logger.info("Message processed successfully")
        except Exception as history_error:
            logger.error(f"Error updating chat history: {str(history_error)}")
            logger.error(traceback.format_exc())
            # Continue without updating chat history if there's an error
        
    except Exception as e:
        error_msg = f"Unhandled error in message processing: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        await cl.Message(content=f"⚠️ An unexpected error occurred: {str(e)}. Please try again or contact support if the issue persists.").send()


class RobustJsonOutputParser(JsonOutputParser):
    def parse_result(self, result, *, partial=False):
        try:
            return super().parse_result(result, partial=partial)
        except Exception as e:
            # Return a fallback structure with the raw text
            return {
                "answer": result,
                "sources": []
            }