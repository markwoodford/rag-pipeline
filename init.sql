-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the document table (one row per source document)
CREATE TABLE IF NOT EXISTS document (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS document_category_idx ON document (category);
CREATE INDEX IF NOT EXISTS document_file_path_idx ON document (file_path);

-- Create the document_chunk table (one row per chunk)
CREATE TABLE IF NOT EXISTS document_chunk (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS document_chunk_document_idx ON document_chunk (document_id);

-- Create an index for cosine similarity search
CREATE INDEX IF NOT EXISTS document_chunk_embedding_idx ON document_chunk
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

