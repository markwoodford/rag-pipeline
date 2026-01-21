/**
 * Embedding Generation Module
 *
 * Generates embeddings for text chunks using AWS Bedrock Titan Embeddings model.
 * The Titan Text Embeddings V2 model produces 1024-dimensional vectors.
 */

import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import type { Chunk } from "./chunk";

/**
 * Represents a chunk with its embedding vector
 */
export interface EmbeddedChunk extends Chunk {
  /** The embedding vector (1024 dimensions for Titan) */
  embedding: number[];
}

/**
 * Configuration for the embedding service
 */
export interface EmbedConfig {
  /** AWS region for Bedrock */
  region: string;
  /** AWS access key ID */
  accessKeyId: string;
  /** AWS secret access key */
  secretAccessKey: string;
  /** Bedrock Titan embedding model ID */
  modelId: string;
}

/**
 * Response from Titan Embeddings model
 */
interface TitanEmbeddingResponse {
  embedding: number[];
  inputTextTokenCount: number;
}

/**
 * Get embedding configuration from environment variables
 */
export function getEmbedConfig(): EmbedConfig {
  const region = process.env.AWS_REGION || "us-east-1";
  const accessKeyId = process.env.AWS_ACCESS_KEY_ID || "";
  const secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY || "";
  const modelId =
    process.env.BEDROCK_TITAN_EMBEDDING_MODEL_ID ||
    "amazon.titan-embed-text-v2:0";

  return {
    region,
    accessKeyId,
    secretAccessKey,
    modelId,
  };
}

/**
 * Create a Bedrock Runtime client
 */
export function createBedrockClient(config: EmbedConfig): BedrockRuntimeClient {
  return new BedrockRuntimeClient({
    region: config.region,
    credentials: {
      accessKeyId: config.accessKeyId,
      secretAccessKey: config.secretAccessKey,
    },
  });
}

/**
 * Generate embedding for a single text string using Titan Embeddings
 */
export async function generateEmbedding(
  client: BedrockRuntimeClient,
  text: string,
  modelId: string,
): Promise<number[]> {
  // Titan Embeddings V2 request body format
  const requestBody = {
    inputText: text,
    dimensions: 1024,
    normalize: true,
  };

  const command = new InvokeModelCommand({
    modelId,
    contentType: "application/json",
    accept: "application/json",
    body: JSON.stringify(requestBody),
  });

  const response = await client.send(command);

  // Parse response body
  const responseBody = JSON.parse(
    new TextDecoder().decode(response.body),
  ) as TitanEmbeddingResponse;

  return responseBody.embedding;
}

/**
 * Generate embeddings for multiple chunks
 * Processes chunks sequentially to avoid rate limiting
 */
export async function embedChunks(chunks: Chunk[]): Promise<EmbeddedChunk[]> {
  const config = getEmbedConfig();
  const client = createBedrockClient(config);
  const embeddedChunks: EmbeddedChunk[] = [];

  for (const chunk of chunks) {
    const embedding = await generateEmbedding(
      client,
      chunk.content,
      config.modelId,
    );
    embeddedChunks.push({
      ...chunk,
      embedding,
    });
  }

  return embeddedChunks;
}
