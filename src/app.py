import os
from typing import List
import dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_pinecone import Pinecone
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from chainlit.input_widget import Select, Switch, Slider
import re
import chainlit as cl
dotenv.load_dotenv()

pdf_data = {
    "WHO guideline on control and elimination of human schistosomiasis (2022)":"https://iris.who.int/bitstream/handle/10665/351856/9789240041608-eng.pdf",
    "WHO guidelines for malaria (2023)":"https://iris.who.int/bitstream/handle/10665/373339/WHO-UCN-GMP-2023.01-Rev.1-eng.pdf"
}

@cl.on_chat_start
async def on_chat_start():

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Choose Model",
                values=["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "gpt-3.5-turbo", 'llama3-70b-8192', 'llama3-8b-8192','mixtral-8x7b-32768', 'gemma-7b-it'],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Model - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            )]).send()
    await setup_agent(settings)


    # Let the user know that the system is ready
    await cl.Message(content="Hello, I am here to help you with questions on your provided PDF files.").send()



@cl.on_settings_update
async def setup_agent(settings):
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings,
                                             namespace=os.getenv("PINECONE_NAME_SPACE"))
    retriever = docsearch.as_retriever(search_type="similarity")
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    if (settings['model'] == "gpt-3.5-turbo"):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = settings['temperature'], streaming=True)
    else:
        llm = ChatGroq(model_name=settings['model'], temperature = settings['temperature'], streaming= True, api_key=os.getenv("GROQ_API_KEY"))
        # Create a chain that uses the Pinecone vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]
    print(source_documents)
    elements = []  # type: List[cl.Element]

    if source_documents:
        for source_doc in source_documents:
            page_num = source_doc.metadata.get("page", "")
            pdf_title = source_doc.metadata.get("source", "")
            # Create a text element for the source citation
            elements.append(
                cl.Text(
                    content=source_doc.page_content,
                    name=f"page_{page_num}",
                )
            )

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
            answer += "\n" + "\n" + "Sources:\n" + "\n".join(source_citations)
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=elements).send()