import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
import dotenv
dotenv.load_dotenv()



file_path = '../pdf_data'

directory_loader = DirectoryLoader(file_path, loader_cls=PyPDFLoader)
raw_docs = directory_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(raw_docs)

print('split docs', docs)
print('creating vector store...')

embeddings = OpenAIEmbeddings()

PineconeVectorStore.from_documents(docs, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"), namespace=os.getenv('PINECONE_NAME_SPACE'))
print('Data ingestion finished')