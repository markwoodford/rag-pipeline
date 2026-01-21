/**
 * Database Configuration and Operations
 *
 * Provides configuration, connection pool management, and CRUD operations
 * for the PostgreSQL database with PgVector.
 */

import { Pool, PoolClient, QueryResult, QueryResultRow } from "pg";
import pgvector from "pgvector/pg";

/**
 * Database configuration interface
 */
export interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
}

/**
 * Document record as stored in the database
 */
export interface DocumentRecord {
  id: number;
  content: string;
  category: string;
  filePath: string;
  createdAt: Date;
}

/**
 * Document chunk record as stored in the database
 */
export interface DocumentChunkRecord {
  id: number;
  documentId: number;
  content: string;
  chunkIndex: number;
  startOffset: number;
  endOffset: number;
  embedding: number[];
  createdAt: Date;
}

/**
 * Input for inserting a document
 */
export interface DocumentInput {
  content: string;
  category: string;
  filePath: string;
}

/**
 * Input for inserting a document chunk
 */
export interface DocumentChunkInput {
  content: string;
  chunkIndex: number;
  startOffset: number;
  endOffset: number;
  embedding: number[];
}

/**
 * Input for inserting a document with its chunks
 */
export interface DocumentWithChunksInput {
  document: DocumentInput;
  chunks: DocumentChunkInput[];
}

/**
 * Search result with similarity score
 */
export interface SearchResult {
  id: number;
  documentId: number;
  content: string;
  category: string;
  filePath: string;
  chunkIndex: number;
  startOffset: number;
  endOffset: number;
  similarity: number;
  rank?: number;
}

// Module-level pool instance
let pool: Pool | null = null;

/**
 * Get database configuration from environment variables
 */
export function getDatabaseConfig(): DatabaseConfig {
  return {
    host: process.env.POSTGRES_HOST || "localhost",
    port: parseInt(process.env.POSTGRES_PORT || "5432", 10),
    database: process.env.POSTGRES_DB || "rag_pipeline",
    user: process.env.POSTGRES_USER || "postgres",
    password: process.env.POSTGRES_PASSWORD || "postgres",
  };
}

/**
 * Get database connection string
 */
export function getDatabaseConnectionString(): string {
  const config = getDatabaseConfig();
  return `postgresql://${config.user}:${config.password}@${config.host}:${config.port}/${config.database}`;
}

/**
 * Create a new database connection pool
 */
export function createPool(): Pool {
  const config = getDatabaseConfig();

  return new Pool({
    host: config.host,
    port: config.port,
    database: config.database,
    user: config.user,
    password: config.password,
    max: 10,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
  });
}

/**
 * Get or create the singleton pool instance
 */
export function getPool(): Pool {
  if (!pool) {
    pool = createPool();
  }
  return pool;
}

/**
 * Close the pool connection
 */
export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
  }
}

/**
 * Register pgvector type with a client
 */
export async function registerVectorType(client: PoolClient): Promise<void> {
  await pgvector.registerType(client);
}

/**
 * Execute a query with the pool
 */
export async function query<T extends QueryResultRow = QueryResultRow>(
  sql: string,
  params?: unknown[],
): Promise<QueryResult<T>> {
  const poolInstance = getPool();
  return poolInstance.query<T>(sql, params);
}

/**
 * Get a client from the pool for transaction support
 */
export async function getClient(): Promise<PoolClient> {
  const poolInstance = getPool();
  const client = await poolInstance.connect();
  await registerVectorType(client);
  return client;
}

/**
 * Insert a single document into the database
 */
export async function insertDocument(
  doc: DocumentInput,
  client?: PoolClient,
): Promise<number> {
  const sql = `
    INSERT INTO document (content, category, file_path)
    VALUES ($1, $2, $3)
    RETURNING id
  `;

  const params = [doc.content, doc.category, doc.filePath];

  if (client) {
    const result = await client.query<{ id: number }>(sql, params);
    return result.rows[0].id;
  }

  const poolInstance = getPool();
  const poolClient = await poolInstance.connect();
  try {
    await registerVectorType(poolClient);
    const result = await poolClient.query<{ id: number }>(sql, params);
    return result.rows[0].id;
  } finally {
    poolClient.release();
  }
}

/**
 * Insert multiple document chunks in a batch
 */
export async function insertDocumentChunks(
  documentId: number,
  chunks: DocumentChunkInput[],
  client?: PoolClient,
): Promise<number[]> {
  if (chunks.length === 0) {
    return [];
  }

  const sql = `
    INSERT INTO document_chunk (document_id, content, chunk_index, start_offset, end_offset, embedding)
    VALUES ($1, $2, $3, $4, $5, $6)
    RETURNING id
  `;

  const effectiveClient = client || (await getClient());
  const insertedIds: number[] = [];

  try {
    for (const chunk of chunks) {
      const params = [
        documentId,
        chunk.content,
        chunk.chunkIndex,
        chunk.startOffset,
        chunk.endOffset,
        pgvector.toSql(chunk.embedding),
      ];
      const result = await effectiveClient.query<{ id: number }>(sql, params);
      insertedIds.push(result.rows[0].id);
    }
    return insertedIds;
  } finally {
    if (!client) {
      effectiveClient.release();
    }
  }
}

/**
 * Insert multiple documents with their chunks in a batch
 */
export async function insertDocuments(
  docs: DocumentWithChunksInput[],
): Promise<number[]> {
  if (docs.length === 0) {
    return [];
  }

  const client = await getClient();
  const ids: number[] = [];

  try {
    await client.query("BEGIN");

    for (const doc of docs) {
      const documentId = await insertDocument(doc.document, client);
      ids.push(documentId);
      await insertDocumentChunks(documentId, doc.chunks, client);
    }

    await client.query("COMMIT");
    return ids;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}

/**
 * Get a document by ID
 */
export async function getDocumentById(
  id: number,
): Promise<DocumentRecord | null> {
  const sql = `
    SELECT
      id,
      content,
      category,
      file_path AS "filePath",
      created_at AS "createdAt"
    FROM document
    WHERE id = $1
  `;

  const client = await getClient();
  try {
    const result = await client.query<DocumentRecord>(sql, [id]);
    if (result.rows.length === 0) {
      return null;
    }
    return result.rows[0];
  } finally {
    client.release();
  }
}

/**
 * Search documents by vector similarity (cosine distance)
 */
export async function searchByVector(
  embedding: number[],
  limit: number = 10,
  category?: string,
): Promise<SearchResult[]> {
  const sql = `
    SELECT 
      dc.id,
      dc.document_id AS "documentId",
      dc.content,
      d.category,
      d.file_path AS "filePath",
      dc.chunk_index AS "chunkIndex",
      dc.start_offset AS "startOffset",
      dc.end_offset AS "endOffset",
      1 - (dc.embedding <=> $1) as similarity
    FROM document_chunk dc
    JOIN document d ON d.id = dc.document_id
    WHERE dc.embedding IS NOT NULL
      AND ($3::text IS NULL OR d.category = $3)
    ORDER BY dc.embedding <=> $1
    LIMIT $2
  `;

  const client = await getClient();
  try {
    const result = await client.query<SearchResult>(sql, [
      pgvector.toSql(embedding),
      limit,
      category ?? null,
    ]);
    return result.rows;
  } finally {
    client.release();
  }
}
