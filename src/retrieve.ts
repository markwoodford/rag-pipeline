/**
 * Retrieve Script
 *
 * Performs vector search (cosine similarity) to retrieve
 * relevant document chunks for a given query.
 *
 * Usage: npm run retrieve -- "<query>"
 */

import { searchByVector, closePool, SearchResult } from "./utils/database";
import {
  getEmbedConfig,
  createBedrockClient,
  generateEmbedding,
} from "./utils/embed";
import { getQueryFromArgs } from "./utils/cli";

/**
 * Configuration for the retrieval operation
 */
export interface RetrieveConfig {
  /** Number of results to return */
  topK: number;
}

/**
 * Result of a retrieval operation
 */
export interface RetrieveResult {
  /** The original query */
  query: string;
  /** Search results */
  results: SearchResult[];
  /** Time taken in milliseconds */
  timeTaken: number;
}

/**
 * Get retrieval configuration from environment variables
 */
function getRetrieveConfig(): RetrieveConfig {
  const topK = parseInt(process.env.RETRIEVAL_TOP_K || "5", 10);

  return {
    topK,
  };
}

/**
 * Format search results for display
 */
function formatResults(results: SearchResult[]): string {
  if (results.length === 0) {
    return "No results found.";
  }

  const lines: string[] = [];
  lines.push(`Found ${results.length} result(s):\n`);

  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    lines.push(`--- Result ${i + 1} ---`);
    lines.push(`File path: ${result.filePath}`);
    lines.push(`Chunk: ${result.chunkIndex + 1}`);
    lines.push(`Similarity: ${result.similarity.toFixed(4)}`);

    // Show a preview of the content (first 200 chars)
    const contentPreview =
      result.content.length > 200
        ? result.content.substring(0, 200) + "..."
        : result.content;
    lines.push(`Content:\n${contentPreview}`);
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Main retrieval function
 * Returns the full retrieval result including results and timing
 */
export async function retrieve(
  query: string,
): Promise<RetrieveResult> {
  const startTime = Date.now();

  // Get configuration
  const retrieveConfig = getRetrieveConfig();

  // Generate embedding for the query
  const embedConfig = getEmbedConfig();
  const client = createBedrockClient(embedConfig);
  const embedding = await generateEmbedding(client, query, embedConfig.modelId);

  // Perform the retrieval
  const results = await searchByVector(embedding, retrieveConfig.topK);

  const timeTaken = Date.now() - startTime;

  return {
    query,
    results,
    timeTaken,
  };
}

async function main(): Promise<void> {
  try {
    // Get query from command line
    const query = getQueryFromArgs(process.argv);

    if (!query) {
      console.error('Usage: npm run retrieve -- "<query>"');
      console.error(
        'Example: npm run retrieve -- "How do I configure the database?"',
      );
      process.exit(1);
    }

    console.log(`Searching for: "${query}"`);
    console.log("");

    // Get configuration
    const config = getRetrieveConfig();
    console.log(`Top K: ${config.topK}`);
    console.log("");

    // Perform retrieval
    const result = await retrieve(query);

    // Display results
    console.log(formatResults(result.results));
    console.log(`Search completed in ${result.timeTaken}ms`);
  } catch (error) {
    console.error(
      "Error during retrieval:",
      error instanceof Error ? error.message : error,
    );
    process.exit(1);
  } finally {
    // Clean up database connection
    await closePool();
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}
