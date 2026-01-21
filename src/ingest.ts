/**
 * Ingest Script
 *
 * Reads markdown files from a directory, chunks them, creates embeddings,
 * and inserts them into the database.
 *
 * Usage: npm run ingest
 *
 * Requires:
 *   DOCS_DIRECTORY env var pointing at the docs folder
 */

import * as fs from "fs";
import * as path from "path";
import { chunkDocuments, getChunkConfig } from "./utils/chunk";
import { embedChunks, EmbeddedChunk } from "./utils/embed";
import {
  closePool,
  DocumentWithChunksInput,
  insertDocuments,
} from "./utils/database";

/**
 * Represents a document loaded from a markdown file
 */
export interface Document {
  /** The file path relative to the docs directory */
  filePath: string;
  /** The full content of the markdown file */
  content: string;
  /** Category (directory name) */
  category: string;
}

/**
 * Check if a file has a markdown extension
 */
function isMarkdownFile(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  return ext === ".md" || ext === ".markdown";
}

/**
 * Recursively find all markdown files in a directory
 */
function findMarkdownFiles(directory: string): string[] {
  const markdownFiles: string[] = [];

  if (!fs.existsSync(directory)) {
    return markdownFiles;
  }

  const entries = fs.readdirSync(directory, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);

    if (entry.isDirectory()) {
      // Recursively search subdirectories
      markdownFiles.push(...findMarkdownFiles(fullPath));
    } else if (entry.isFile() && isMarkdownFile(entry.name)) {
      markdownFiles.push(fullPath);
    }
  }

  return markdownFiles;
}

/**
 * Extract the category (directory name) from a file path
 * Files in the root docs folder get category "general"
 */
function extractCategory(relativePath: string): string {
  const dir = path.dirname(relativePath);
  if (dir === "." || dir === "") {
    return "general";
  }
  // Use the first directory component as the category
  const parts = dir.split(path.sep);
  return parts[0];
}

/**
 * Load a markdown file and return a Document object
 */
function loadMarkdownFile(
  filePath: string,
  baseDirectory: string,
): Document {
  const content = fs.readFileSync(filePath, "utf-8");
  const relativePath = path.relative(baseDirectory, filePath);
  const category = extractCategory(relativePath);

  return {
    filePath: relativePath,
    content,
    category,
  };
}

/**
 * Load all markdown files from a directory
 */
function loadDocuments(directory: string): Document[] {
  const absoluteDir = path.resolve(directory);
  const markdownFiles = findMarkdownFiles(absoluteDir);

  return markdownFiles.map((filePath) =>
    loadMarkdownFile(filePath, absoluteDir),
  );
}

/**
 * Get the documents directory from the environment variable
 */
function getDocsDirectory(): string {
  const docsDirectory = process.env.DOCS_DIRECTORY;
  if (!docsDirectory) {
    throw new Error("DOCS_DIRECTORY env var is required to run ingest.");
  }
  return docsDirectory;
}

/**
 * Main ingest function
 */
function buildDocumentInputs(
  documents: Document[],
  embeddedChunks: EmbeddedChunk[],
): DocumentWithChunksInput[] {
  const chunksByFilePath = new Map<string, EmbeddedChunk[]>();

  for (const chunk of embeddedChunks) {
    const filePath = chunk.metadata.filePath;
    const existing = chunksByFilePath.get(filePath);
    if (existing) {
      existing.push(chunk);
    } else {
      chunksByFilePath.set(filePath, [chunk]);
    }
  }

  return documents
    .map((document) => {
      const docChunks = chunksByFilePath.get(document.filePath) || [];
      return {
        document: {
          content: document.content,
          category: document.category,
          filePath: document.filePath,
        },
        chunks: docChunks.map((chunk) => ({
          content: chunk.content,
          chunkIndex: chunk.metadata.chunkIndex,
          startOffset: chunk.metadata.startOffset,
          endOffset: chunk.metadata.endOffset,
          embedding: chunk.embedding,
        })),
      };
    })
    .filter((doc) => doc.chunks.length > 0);
}

export async function ingest(): Promise<Document[]> {
  const docsDir = getDocsDirectory();
  const absolutePath = path.resolve(docsDir);

  console.log(`Ingesting documents from: ${absolutePath}`);

  if (!fs.existsSync(absolutePath)) {
    console.error(`Error: Directory does not exist: ${absolutePath}`);
    return [];
  }

  const documents = loadDocuments(absolutePath);

  console.log(`Found ${documents.length} markdown file(s)`);

  for (const doc of documents) {
    console.log(`  - ${doc.filePath}`);
  }

  if (documents.length === 0) {
    return [];
  }

  const chunkConfig = getChunkConfig();

  console.log(
    `Chunking documents (size=${chunkConfig.chunkSize}, overlap=${chunkConfig.chunkOverlap})...`,
  );
  const chunks = await chunkDocuments(documents);
  console.log(`Generated ${chunks.length} chunk(s)`);

  if (chunks.length === 0) {
    console.log("No chunks generated, skipping embedding and database insert.");
    return documents;
  }

  console.log(`Generating embeddings for ${chunks.length} chunks...`);
  const embeddedChunks = await embedChunks(chunks);
  console.log(`Successfully generated ${embeddedChunks.length} embeddings`);

  const docsToInsert = buildDocumentInputs(documents, embeddedChunks);

  const insertedIds = await insertDocuments(docsToInsert);
  const totalChunks = docsToInsert.reduce(
    (sum, doc) => sum + doc.chunks.length,
    0,
  );
  console.log(
    `Inserted ${insertedIds.length} document(s) and ${totalChunks} chunk(s) into the database`,
  );

  return documents;
}

// Run if executed directly
if (require.main === module) {
  ingest()
    .then((documents) => {
      if (documents.length === 0) {
        console.log("No documents found to ingest.");
        process.exit(1);
      }
      console.log("Ingestion complete.");
    })
    .catch((error) => {
      console.error("Ingestion failed:", error);
      process.exit(1);
    })
    .finally(async () => {
      await closePool();
    });
}
