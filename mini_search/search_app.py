import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "and", "or", "with", "what", "how", "why", "does", "do", "it"
}


@dataclass
class SearchResult:
    source: str
    text: str
    score: int


def tokenize(text: str) -> List[str]:
    """Convert text into lowercase keyword tokens."""
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [token for token in raw_tokens if token not in STOPWORDS]


def score_document(query: str, document_text: str) -> int:
    """Score a document by counting overlapping unique keywords."""
    query_tokens = set(tokenize(query))
    document_tokens = set(tokenize(document_text))
    return len(query_tokens.intersection(document_tokens))


def load_documents(docs_dir: str = "docs") -> Dict[str, str]:
    """Load all .txt documents from a directory."""
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        return {}

    documents = {}

    for file_path in sorted(docs_path.glob("*.txt")):
        documents[str(file_path)] = file_path.read_text(encoding="utf-8").strip()

    return documents


def retrieve(query: str, documents: Dict[str, str], top_k: int = 2) -> List[SearchResult]:
    """Return the top-k documents with a positive relevance score."""
    results = []

    for source, text in documents.items():
        score = score_document(query, text)

        if score > 0:
            results.append(SearchResult(source=source, text=text, score=score))

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:top_k]

def first_sentence(text: str) -> str:
    """Return the first sentence from a document."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[0] if sentences else text


def build_answer(query: str, results: List[SearchResult]) -> str:
    """Build a simple grounded answer using retrieved documents."""
    if not results:
        return "Answer:\nI don't have enough information.\n\nSources:\n- None"

    best_result = results[0]

    answer = (
        "Answer:\n"
        f"{first_sentence(best_result.text)}\n\n"
        "Sources:\n"
    )

    for result in results:
        answer += f"- {result.source}\n"

    return answer.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny grounded local search CLI.")
    parser.add_argument("query", help="Question or search query")
    parser.add_argument("--docs-dir", default="docs", help="Directory containing .txt documents")
    parser.add_argument("--top-k", type=int, default=2, help="Number of documents to retrieve")

    args = parser.parse_args()

    documents = load_documents(args.docs_dir)
    results = retrieve(args.query, documents, args.top_k)
    answer = build_answer(args.query, results)

    print(answer)


if __name__ == "__main__":
    main()
