/**
 * Chunking Module
 *
 * Splits markdown documents into chunks using LangChain's RecursiveCharacterTextSplitter.
 */

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import type { Document } from "../ingest";

/**
 * Represents a chunk of text from a document
 */
export interface Chunk {
  /** The text content of the chunk */
  content: string;
  /** Metadata about the chunk */
  metadata: {
    /** File path */
    filePath: string;
    /** Index of this chunk within the document */
    chunkIndex: number;
    /** Character offset where this chunk starts in the original document */
    startOffset: number;
    /** Character offset where this chunk ends in the original document */
    endOffset: number;
  };
}

/**
 * Configuration for chunking
 */
export interface ChunkConfig {
  /** Maximum size of each chunk in characters */
  chunkSize: number;
  /** Number of characters to overlap between chunks */
  chunkOverlap: number;
}

/**
 * Get chunking configuration from environment variables
 */
export function getChunkConfig(): ChunkConfig {
  const chunkSize = parseInt(process.env.CHUNK_SIZE || "500", 10);
  const chunkOverlap = parseInt(process.env.CHUNK_OVERLAP || "100", 10);

  return { chunkSize, chunkOverlap };
}

function findChunkOffsets(
  content: string,
  chunks: string[],
  chunkOverlap: number,
): Array<{ startOffset: number; endOffset: number }> {
  const offsets: Array<{ startOffset: number; endOffset: number }> = [];
  let previousEnd = 0;

  for (const chunk of chunks) {
    const searchStart = Math.max(0, previousEnd - chunkOverlap);
    let startOffset = content.indexOf(chunk, searchStart);

    if (startOffset === -1 && searchStart > 0) {
      startOffset = content.indexOf(chunk);
    }

    if (startOffset === -1) {
      startOffset = searchStart;
    }

    const endOffset = Math.min(startOffset + chunk.length, content.length);
    offsets.push({ startOffset, endOffset });
    previousEnd = endOffset;
  }

  return offsets;
}

function createMarkdownSplitter(
  config: ChunkConfig,
): RecursiveCharacterTextSplitter {
  return RecursiveCharacterTextSplitter.fromLanguage("markdown", {
    chunkSize: config.chunkSize,
    chunkOverlap: config.chunkOverlap,
    lengthFunction: (text: string) => text.length,
  });
}

/**
 * Split a single document into chunks
 */
export async function chunkDocument(
  document: Document,
  config: ChunkConfig,
  splitter: RecursiveCharacterTextSplitter = createMarkdownSplitter(config),
): Promise<Chunk[]> {
  const { chunkOverlap } = config;
  const { content, filePath } = document;

  // Handle empty documents or documents with only whitespace
  const trimmedContent = content.trim();
  if (trimmedContent.length === 0) {
    return [];
  }

  const rawSplits = await splitter.splitText(content);
  const splits = rawSplits.filter((split: string) => split.trim().length > 0);
  if (splits.length === 0) {
    return [];
  }

  const offsets = findChunkOffsets(content, splits, chunkOverlap);

  const chunks: Chunk[] = [];
  let chunkIndex = 0;

  for (let i = 0; i < splits.length; i++) {
    const rawSplit = splits[i];
    const trimmedSplit = rawSplit.trim();
    if (trimmedSplit.length === 0) {
      continue;
    }

    const rawOffsets = offsets[i];
    const leadingTrim = rawSplit.length - rawSplit.trimStart().length;
    const trailingTrim = rawSplit.length - rawSplit.trimEnd().length;
    const startOffset = Math.min(
      rawOffsets.startOffset + leadingTrim,
      content.length,
    );
    const endOffset = Math.min(
      rawOffsets.endOffset - trailingTrim,
      content.length,
    );
    const safeEndOffset = Math.max(startOffset, endOffset);

    chunks.push({
      content: trimmedSplit,
      metadata: {
        filePath,
        chunkIndex,
        startOffset,
        endOffset: safeEndOffset,
      },
    });
    chunkIndex += 1;
  }

  return chunks;
}

/**
 * Chunk multiple documents
 */
export async function chunkDocuments(documents: Document[]): Promise<Chunk[]> {
  const config = getChunkConfig();
  const splitter = createMarkdownSplitter(config);
  const allChunks: Chunk[] = [];

  for (const document of documents) {
    const chunks = await chunkDocument(document, config, splitter);
    allChunks.push(...chunks);
  }

  return allChunks;
}
