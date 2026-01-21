# RAG Pipeline

A simple Retrieval-Augmented Generation (RAG) pipeline written in TypeScript, using PostgreSQL + pgvector
for storage and AWS Bedrock for embeddings and generation.

Chunking of docs is done using the recursive text splitter from LangChain.

## Prerequisites

- Docker
- AWS Bedrock credentials

## Setup

1. Copy the example environment file and fill in your values: `cp example.env .env`
2. Ensure `DOCS_DIRECTORY` points to your markdown docs folder (default: `docs`).

## Build and run (Docker)

- Start the database:
  - `./rag-pipeline.sh up`
- Build the app image:
  - `./rag-pipeline.sh build`

## Run the pipeline

- Ingest documents:
  - `./rag-pipeline.sh ingest`
- Retrieve context:
  - `./rag-pipeline.sh retrieve "your query"`
- Generate an answer:
  - `./rag-pipeline.sh generate "your query"`

## Configuration

Common environment variables (in `.env`):

- `DOCS_DIRECTORY` - path to markdown docs (relative to repo root)
- `CHUNK_SIZE` / `CHUNK_OVERLAP` - chunking config in characters
- `RETRIEVAL_TOP_K` - number of chunks to return for retrieval
- `AWS_*` / `BEDROCK_*` - Bedrock credentials and model IDs
- `POSTGRES_*` - database connection settings
