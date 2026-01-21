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
