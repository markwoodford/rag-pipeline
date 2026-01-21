/**
 * Generate Script
 *
 * Uses Claude 4.5 Sonnet with retrieved context to generate responses
 * to user queries using a RAG (Retrieval-Augmented Generation) approach.
 *
 * Usage: npm run generate -- "<query>"
 */

import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { retrieve, RetrieveResult } from "./retrieve";
import { closePool, SearchResult } from "./utils/database";
import { createBedrockClient } from "./utils/embed";
import { getQueryFromArgs } from "./utils/cli";

/**
 * Configuration for the generation operation
 */
interface GenerateConfig {
  /** AWS region for Bedrock */
  region: string;
  /** AWS access key ID */
  accessKeyId: string;
  /** AWS secret access key */
  secretAccessKey: string;
  /** Bedrock Claude model ID */
  modelId: string;
  /** Maximum tokens to generate */
  maxTokens: number;
  /** Temperature for generation (0-1) */
  temperature: number;
}

/**
 * Result of a generation operation
 */
interface GenerateResult {
  /** The original query */
  query: string;
  /** The generated response */
  response: string;
  /** Retrieved context chunks used */
  context: SearchResult[];
  /** Number of context chunks used */
  contextCount: number;
  /** Time taken for retrieval in milliseconds */
  retrievalTime: number;
  /** Time taken for generation in milliseconds */
  generationTime: number;
  /** Total time taken in milliseconds */
  totalTime: number;
}

/**
 * Claude message response structure
 */
interface ClaudeResponse {
  id: string;
  type: string;
  role: string;
  content: Array<{
    type: string;
    text: string;
  }>;
  model: string;
  stop_reason: string;
  stop_sequence: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

/**
 * Get generation configuration from environment variables
 */
function getGenerateConfig(): GenerateConfig {
  const region = process.env.AWS_REGION || "us-east-1";
  const accessKeyId = process.env.AWS_ACCESS_KEY_ID || "";
  const secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY || "";
  const modelId =
    process.env.BEDROCK_CLAUDE_MODEL_ID ||
    "anthropic.claude-sonnet-4-5-20250115-v1:0";
  const maxTokens = parseInt(process.env.GENERATE_MAX_TOKENS || "1024", 10);
  const temperature = parseFloat(process.env.GENERATE_TEMPERATURE || "0.7");

  return {
    region,
    accessKeyId,
    secretAccessKey,
    modelId,
    maxTokens,
    temperature,
  };
}

/**
 * Format retrieved context for the prompt
 */
function formatContextForPrompt(results: SearchResult[]): string {
  if (results.length === 0) {
    return "No relevant context found.";
  }

  const contextParts: string[] = [];

  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    contextParts.push(`[Document ${i + 1}] (Source: ${result.filePath})`);
    contextParts.push(result.content);
    contextParts.push(""); // Empty line between documents
  }

  return contextParts.join("\n");
}

/**
 * Build the system prompt for RAG generation
 */
function buildSystemPrompt(): string {
  return `You are a helpful assistant that answers questions based on the provided documentation context. 

Your task is to:
1. Read the provided context carefully
2. Answer the user's question based ONLY on the information in the context
3. If the context doesn't contain relevant information to answer the question, say so clearly
4. Cite the source documents when providing information
5. Be concise and direct in your responses

Do not make up information or use knowledge outside of the provided context.`;
}

/**
 * Build the user prompt with query and context
 */
function buildUserPrompt(query: string, context: string): string {
  return `Context from documentation:
---
${context}
---

Question: ${query}

Please answer the question based on the context provided above.`;
}

/**
 * Generate a response using Claude via Bedrock
 */
async function generateResponse(
  client: BedrockRuntimeClient,
  query: string,
  context: SearchResult[],
  config: GenerateConfig,
): Promise<string> {
  const formattedContext = formatContextForPrompt(context);
  const systemPrompt = buildSystemPrompt();
  const userPrompt = buildUserPrompt(query, formattedContext);

  // Claude Messages API request body
  const requestBody = {
    anthropic_version: "bedrock-2023-05-31",
    max_tokens: config.maxTokens,
    temperature: config.temperature,
    system: systemPrompt,
    messages: [
      {
        role: "user",
        content: userPrompt,
      },
    ],
  };

  const command = new InvokeModelCommand({
    modelId: config.modelId,
    contentType: "application/json",
    accept: "application/json",
    body: JSON.stringify(requestBody),
  });

  const response = await client.send(command);

  // Parse response body
  const responseBody = JSON.parse(
    new TextDecoder().decode(response.body),
  ) as ClaudeResponse;

  // Extract text from response
  if (responseBody.content && responseBody.content.length > 0) {
    return responseBody.content
      .filter((block) => block.type === "text")
      .map((block) => block.text)
      .join("\n");
  }

  return "";
}

/**
 * Main generation function
 * Retrieves context and generates a response
 */
export async function generate(
  query: string,
): Promise<GenerateResult> {
  const startTime = Date.now();

  // Get generation configuration
  const config = getGenerateConfig();

  // Step 1: Retrieve relevant context
  const retrieveStartTime = Date.now();
  let retrieveResult: RetrieveResult;

  try {
    retrieveResult = await retrieve(query);
  } catch (error) {
    // If retrieval fails, we can still try to generate with no context
    console.warn(
      "Warning: Retrieval failed, generating response without context",
    );
    retrieveResult = {
      query,
      results: [],
      timeTaken: Date.now() - retrieveStartTime,
    };
  }

  const retrievalTime = Date.now() - retrieveStartTime;

  // Step 2: Generate response with context
  const generationStartTime = Date.now();
  const client = createBedrockClient(config);
  const response = await generateResponse(
    client,
    query,
    retrieveResult.results,
    config,
  );
  const generationTime = Date.now() - generationStartTime;

  const totalTime = Date.now() - startTime;

  return {
    query,
    response,
    context: retrieveResult.results,
    contextCount: retrieveResult.results.length,
    retrievalTime,
    generationTime,
    totalTime,
  };
}

/**
 * Format and display the generation result
 */
function formatGenerateResult(result: GenerateResult): string {
  const lines: string[] = [];

  lines.push("=".repeat(60));
  lines.push("RAG GENERATION RESULT");
  lines.push("=".repeat(60));
  lines.push("");
  lines.push(`Query: ${result.query}`);
  lines.push("");
  lines.push("-".repeat(60));
  lines.push("RESPONSE:");
  lines.push("-".repeat(60));
  lines.push(result.response);
  lines.push("");
  lines.push("-".repeat(60));
  lines.push("STATISTICS:");
  lines.push("-".repeat(60));
  lines.push(`Context chunks used: ${result.contextCount}`);
  lines.push(`Retrieval time: ${result.retrievalTime}ms`);
  lines.push(`Generation time: ${result.generationTime}ms`);
  lines.push(`Total time: ${result.totalTime}ms`);
  lines.push("");

  if (result.contextCount > 0) {
    lines.push("-".repeat(60));
    lines.push("SOURCES:");
    lines.push("-".repeat(60));
    const uniqueSources = [...new Set(result.context.map((c) => c.filePath))];
    uniqueSources.forEach((source, index) => {
      lines.push(`${index + 1}. ${source}`);
    });
  }

  lines.push("=".repeat(60));

  return lines.join("\n");
}

async function main(): Promise<void> {
  try {
    // Get query from command line
    const query = getQueryFromArgs(process.argv);

    if (!query) {
      console.error('Usage: npm run generate -- "<query>"');
      console.error(
        'Example: npm run generate -- "How do I configure the database?"',
      );
      process.exit(1);
    }

    console.log("Starting RAG generation...");
    console.log(`Query: "${query}"`);
    console.log("");
    console.log("Retrieving relevant context...");

    // Perform generation
    const result = await generate(query);

    // Display results
    console.log("");
    console.log(formatGenerateResult(result));
  } catch (error) {
    console.error(
      "Error during generation:",
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
