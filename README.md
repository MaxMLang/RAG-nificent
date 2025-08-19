# RAG-nificent: An AI Chatbot Powered by LLMs for Citation of Custom PDFs, Reports, and Guidelines
**Disclaimer:** This project is maintained on a best-effort basis as I am not actively using it. APIs may change, and some models have been decommissioned. Please refer to the demo video for the most current optimal use case.

## Now Supports Latest GPT-5, GPT-4.1, and Advanced LLM Models

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-green.svg)](https://langchain.com/)
[![Chainlit](https://img.shields.io/badge/Chainlit-UI-purple.svg)](https://chainlit.io/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-orange.svg)](https://www.pinecone.io/)

<p align="center">
  <img src="https://github.com/MaxMLang/RAG-nificent/raw/master/assets/logo.png" alt="RAG-nificent Logo" width="200" height="200">
</p>
RAG-nificent is a state-of-the-art repository that leverages the power of Retrieval-Augmented Generation (RAG) to provide instant answers and references from a curated directory of PDFs containing information on any given topic such as WHO recommendations documents. This system is designed to aid researchers, policy makers, and the public in quickly finding specific information within extensive documents.

## Features

- **Conversational Interface**: Engage with the system using natural language queries to receive responses directly sourced from the PDFs.
- **Direct Citation**: Every response from the system includes a direct link to the source PDF page, ensuring traceability and verification.
- **PDF Directory**: A predefined set of key PDF documents, currently including WHO recommendations on major health topics such as schistosomiasis and malaria.

## Available Models

### OpenAI Models
- 🚀 **gpt-5-mini-2025-08-07**: Latest GPT-5 model with advanced capabilities
- 🚀 **gpt-5-nano-2025-08-07**: GPT-5 nano version for efficient processing
- 📘 **gpt-4.1-2025-04-14**: GPT-4.1 full version with enhanced reasoning
- 📘 **gpt-4.1-mini-2025-04-14**: GPT-4.1 mini version for cost-effective use
- 💬 **chatgpt-4o-latest**: Latest ChatGPT-4o with improved conversational abilities
- ⚡ **gpt-4-turbo-2024-04-09**: GPT-4 Turbo optimized for complex reasoning
- 💎 **gpt-4o-mini-2024-07-18**: GPT-4o mini for efficient general-purpose tasks

### Groq Models
- 🦙 **llama-3.1-8b-instant**: Fast 8B model for quick interactions
- 🦙 **llama-3.3-70b-versatile**: Versatile 70B model with enhanced capabilities
- 🔬 **deepseek-r1-distill-llama-70b**: Distilled 70B model for efficient processing
- 🌐 **qwen/qwen3-32b**: Qwen 32B model for multilingual tasks

## Demo

https://github.com/user-attachments/assets/d6b1426d-b2f9-4bea-b8fb-bc63facd4724



## How It Works

The application utilizes a combination of OpenAI embeddings, Pinecone vector search, and a conversational interface to provide a seamless retrieval experience. When a query is made, the system:

1. Converts the query into embeddings.
2. Searches for the most relevant document sections using Pinecone's vector search.
3. Returns the answer along with citations and links to the source documents.

## Setup

### Option 1: Standard Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG-nificent.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables in a `.env` (also see `.env.example`file:
   - `PINECONE_INDEX_NAME`
   - `PINECONE_NAME_SPACE`
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`

4. Create a [Pinecone](pinecone.io) index with the same name as  `PINECONE_INDEX_NAME`. Set it up with `dimensions=1536` and `metric=cosine`.
5. Place your PDFs in the `pdf_data` directory and run `data_ingestion.py`
6. Run the application:
   ```bash
   chainlit run src/app.py
   ```

### Option 2: Docker Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG-nificent.git
   ```

2. Create a `.env` file with the required environment variables (see `.env.example`):
   - `PINECONE_INDEX_NAME`
   - `PINECONE_NAME_SPACE`
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`

3. Build and run using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access the application at http://localhost:8000

5. To ingest PDFs with Docker:
   ```bash
   # Place your PDFs in the pdf_data directory first
   docker-compose exec ragnificent python data_ingestion.py
   ```

6. To stop the application:
   ```bash
   docker-compose down
   ```

## Development with Docker

For development purposes, you can use the following commands:

```bash
# Build the Docker image
docker-compose build

# Run the application in development mode (with live reloading)
docker-compose up

# View logs
docker-compose logs -f

# Run a specific command inside the container
docker-compose exec ragnificent <command>
```

## Source Documents

The system currently includes guidelines from the following PDFs with direct links to the documents:

- [WHO guideline on control and elimination of human schistosomiasis (2022)](https://iris.who.int/bitstream/handle/10665/351856/9789240041608-eng.pdf)
- [WHO guidelines for malaria (2023)](https://iris.who.int/bitstream/handle/10665/373339/WHO-UCN-GMP-2023.01-Rev.1-eng.pdf)
