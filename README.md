# RAG-nificent: An AI Chatbot powered by LLMs for Citation of Custom PDFs, Reports, and Guidelines
## Supports Llama-3 by Meta AI
<p align="center">
  <img src="assets/logo.png" alt="Lightning Chatbot Logo" width="200" height="200">
</p>
RAG-nificent is a state-of-the-art repository that leverages the power of Retrieval-Augmented Generation (RAG) to provide instant answers and references from a curated directory of PDFs containing information on any given topic such as WHO recommendations documents. This system is designed to aid researchers, policy makers, and the public in quickly finding specific information within extensive documents.

## Features

- **Conversational Interface**: Engage with the system using natural language queries to receive responses directly sourced from the PDFs.
- **Direct Citation**: Every response from the system includes a direct link to the source PDF page, ensuring traceability and verification.
- **PDF Directory**: A predefined set of key PDF documents, currently including WHO recommendations on major health topics such as schistosomiasis and malaria.

## Available Models
- 📘 **ChatGPT-3.5**: Utilize this advanced iteration of the GPT model for engaging and human-like interactions, suitable for varied conversational tasks.
- 🦙 **Llama3-70B-8192**: Experience high-end performance with this large-scale model, ideal for complex language tasks and deep learning insights.
- 🦙 **Llama3-8B-8192**: Harness robust capabilities with this more accessible version of Llama3, perfect for a wide range of AI applications.
- 🌟 **Mixtral-8x7B-32768**: Leverage the power of ensemble modeling with Mixtral's extensive capacity for nuanced understanding and response generation.
- 🦙 **Llama2-70B-4096**: Utilize the proven effectiveness of Llama2 for comprehensive language processing and application development.
- 💎 **Gemma-7B-IT**: Explore specialized interactions and tech-focused solutions with Gemma, tailored for IT and technical content. 

## Demo

![RAG-nificent Demo](assets/demov0.0.2.gif)

## How It Works

The application utilizes a combination of OpenAI embeddings, Pinecone vector search, and a conversational interface to provide a seamless retrieval experience. When a query is made, the system:

1. Converts the query into embeddings.
2. Searches for the most relevant document sections using Pinecone's vector search.
3. Returns the answer along with citations and links to the source documents.

## Setup

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

## Source Documents

The system currently includes guidelines from the following PDFs with direct links to the documents:

- [WHO guideline on control and elimination of human schistosomiasis (2022)](https://iris.who.int/bitstream/handle/10665/351856/9789240041608-eng.pdf)
- [WHO guidelines for malaria (2023)](https://iris.who.int/bitstream/handle/10665/373339/WHO-UCN-GMP-2023.01-Rev.1-eng.pdf)




