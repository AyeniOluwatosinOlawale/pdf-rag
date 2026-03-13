"""CLI for the PDF RAG system."""

import sys
from rag import PDFRag


def main():
    rag = PDFRag()

    print("PDF RAG — powered by ChromaDB + Claude")
    print("Commands: ingest <path.pdf> | ask <question> | sources | quit\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        elif cmd == "ingest":
            if not arg:
                print("Usage: ingest <path/to/file.pdf>")
            else:
                try:
                    rag.ingest(arg)
                except Exception as e:
                    print(f"Error: {e}")

        elif cmd in ("ask", "query"):
            if not arg:
                print("Usage: ask <your question>")
            else:
                try:
                    answer = rag.query(arg)
                    print(f"\n{answer}\n")
                except Exception as e:
                    print(f"Error: {e}")

        elif cmd == "sources":
            sources = rag.list_sources()
            if sources:
                print("Ingested sources:")
                for s in sources:
                    print(f"  - {s}")
            else:
                print("No PDFs ingested yet.")

        else:
            print(f"Unknown command '{cmd}'. Try: ingest, ask, sources, quit")


if __name__ == "__main__":
    main()
