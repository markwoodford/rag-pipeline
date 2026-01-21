/**
 * Get query from command line arguments
 */
export function getQueryFromArgs(args: string[]): string | null {
  // Skip first two args (node and script path)
  const relevantArgs = args.slice(2);

  if (relevantArgs.length === 0) {
    return null;
  }

  // Join all arguments as the query (in case it's not quoted)
  return relevantArgs.join(" ").trim();
}

export interface ParsedArgs {
  query: string | null;
  category?: string;
}

/**
 * Parse named options and query from command line arguments
 */
export function parseArgs(args: string[]): ParsedArgs {
  // Skip first two args (node and script path)
  const relevantArgs = args.slice(2);
  let category: string | undefined;
  const queryParts: string[] = [];

  for (let i = 0; i < relevantArgs.length; i++) {
    const arg = relevantArgs[i];
    if (arg === "--category" && i + 1 < relevantArgs.length) {
      category = relevantArgs[i + 1];
      i++;
      continue;
    }

    if (!arg.startsWith("--")) {
      queryParts.push(arg);
    }
  }

  return {
    query: queryParts.length > 0 ? queryParts.join(" ").trim() : null,
    category,
  };
}
