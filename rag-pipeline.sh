#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

COMPOSE=()
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "Docker Compose is not installed or not on PATH." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage: ./rag-pipeline.sh <command> [args]

Commands:
  up        Start the database container in background
  down      Stop containers and remove network
  build     Build the app image
  ingest    Ingest docs into the database
  retrieve  Retrieve context for a query
  generate  Generate a response for a query
  eval      Evaluate retrieval using an LLM judge
  logs      Tail database logs
  help      Show this help

Examples:
  ./rag-pipeline.sh up
  ./rag-pipeline.sh ingest
  ./rag-pipeline.sh retrieve "How do I create a packing slip?"
  ./rag-pipeline.sh retrieve "How do I create a packing slip?" --category shipping
  ./rag-pipeline.sh generate "How do I create a packing slip?"
  ./rag-pipeline.sh generate "How do I create a packing slip?" --category shipping
  ./rag-pipeline.sh eval "How do I create a packing slip?"
  ./rag-pipeline.sh eval "How do I create a packing slip?" --category shipping
EOF
}

command="${1:-help}"
shift || true

case "$command" in
  up)
    "${COMPOSE[@]}" up -d db
    ;;
  down)
    "${COMPOSE[@]}" down
    ;;
  build)
    "${COMPOSE[@]}" build app
    ;;
  ingest)
    "${COMPOSE[@]}" run --rm app npm run ingest
    ;;
  retrieve)
    if [ "$#" -eq 0 ]; then
      echo "Query required for retrieve." >&2
      exit 1
    fi
    "${COMPOSE[@]}" run --rm app npm run retrieve -- "$@"
    ;;
  generate)
    if [ "$#" -eq 0 ]; then
      echo "Query required for generate." >&2
      exit 1
    fi
    "${COMPOSE[@]}" run --rm app npm run generate -- "$@"
    ;;
  eval)
    if [ "$#" -eq 0 ]; then
      echo "Query required for eval." >&2
      exit 1
    fi
    "${COMPOSE[@]}" run --rm app npm run eval -- "$@"
    ;;
  logs)
    "${COMPOSE[@]}" logs -f --tail=100 db
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage
    exit 1
    ;;
esac
