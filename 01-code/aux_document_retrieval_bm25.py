import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from typing import Optional
import bm25s

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

@dataclass
class BM25Result:
    rank: int
    doc_id: int
    score: float
    text: Optional[str]
    doc_name: str
    label: str

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "label": self.label,
            "score": self.score,
            "text": self.text
        }


def load_corpus_bm25(corpus_path: Path) -> Tuple[List[str], List[str]]:
    """
    Recursively load text files where filename matches its parent directory name.

    Args:
        corpus_path: Root folder containing subfolders with <name>/<name>.txt files.

    Returns:
        Tuple of:
          - corpus: List of document texts.
          - file_mapping: List of corresponding file paths as strings.
    """
    corpus: List[str] = []
    file_mapping: List[str] = []

    for root, _, files in os.walk(corpus_path):
        parent = Path(root).name
        filename = f"{parent}.txt"
        if filename in files:
            file_path = Path(root) / filename
            try:
                text = file_path.read_text(encoding="utf-8")
                corpus.append(text)
                file_mapping.append(str(file_path))
            except Exception as e:
                logger.warning("Failed to read %s: %s", file_path, e)

    return corpus, file_mapping


def create_bm25_index(
    input_path: Path,
    retriever_path: Path
) -> bm25s.BM25:
    """
    Build, save, and return a BM25 index from a corpus directory.

    Args:
        input_path: Path to folder of documents.
        retriever_path: Base Path for saving the BM25 model and metadata.

    Returns:
        A fitted BM25 retriever instance.
    """
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Invalid input path: {input_path}")

    logger.info("Loading corpus from %s", input_path)
    corpus, file_mapping = load_corpus_bm25(input_path)

    logger.info("Tokenizing %d documents", len(corpus))
    tokens = bm25s.tokenize(corpus, stopwords="en")

    logger.info("Indexing corpus with BM25")
    retriever = bm25s.BM25()
    retriever.index(tokens)

    retriever_path.parent.mkdir(parents=True, exist_ok=True)

    retriever.save(str(retriever_path))
    retriever.save(str(retriever_path) + "_corpus", corpus=corpus)

    mapping_path = retriever_path.with_name(retriever_path.name + "_filenames.json")
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(file_mapping, f)

    logger.info("BM25 index and metadata saved to %s", retriever_path)
    return retriever


def run_bm25_query(
    paths: Dict[str, Path],
    query: str,
    top_k: int = 10,
    vsm_ids: Optional[List[int]] = None
) -> List[dict]:
    """
    Load or build BM25 index, execute query, and return top-k results.
    Optionally limit to a subset of doc IDs.
    """
    logger.info("Running BM25-based query")
    retriever_path = paths["retriever"]

    if not retriever_path.exists():
        retriever = create_bm25_index(paths["pdf_folder"], retriever_path)
    else:
        logger.info("Loading existing BM25 model from %s", retriever_path)
        retriever = bm25s.BM25.load(str(retriever_path), load_corpus=True)

    corpus = getattr(retriever, "corpus", None)

    results = query_bm25(
        retriever_path=retriever_path,
        retriever=retriever,
        query=query,
        k=top_k,
        corpus=corpus,
        vsm_ids=vsm_ids
    )

    return [r.to_dict() for r in results]

def query_bm25(
    retriever_path: Path,
    retriever: bm25s.BM25,
    query: str,
    k: int = 5,
    corpus: Optional[List[str]] = None,
    vsm_ids: Optional[List[int]] = None
) -> List[BM25Result]:
    """
    Query BM25 index and return top-k results as BM25Result instances.
    Optionally restrict to a subset of document indices.
    """
    if not query:
        raise ValueError("Query string is empty.")

    logger.info("Querying BM25 for: '%s'", query)
    q_tokens = bm25s.tokenize(query, stopwords="en")

    mapping_path = retriever_path.with_name(retriever_path.name + "_filenames.json")
    file_names = json.loads(mapping_path.read_text(encoding="utf-8"))

    results: List[BM25Result] = []

    if vsm_ids:
        logger.info("Restricting query to candidate VSM IDs")
        all_scores = retriever.get_scores(q_tokens[0])  # BM25 expects List[str]
        subset = sorted(
            [(doc_id, float(all_scores[doc_id])) for doc_id in vsm_ids],
            key=lambda x: x[1],
            reverse=True
        )[:k]

        for rank, (doc_id, score) in enumerate(subset, start=1):
            path = Path(file_names[doc_id])
            results.append(BM25Result(
                rank=rank,
                doc_id=doc_id,
                score=score,
                text=corpus[doc_id] if corpus else None,
                doc_name=path.parent.name,
                label=path.parent.parent.name
            ))
    else:
        doc_ids, scores = retriever.retrieve(q_tokens, k=k)
        for rank in range(doc_ids.shape[1]):
            doc_id = int(doc_ids[0, rank])
            score = float(scores[0, rank])
            path = Path(file_names[doc_id])
            results.append(BM25Result(
                rank=rank + 1,
                doc_id=doc_id,
                score=score,
                text=corpus[doc_id] if corpus else None,
                doc_name=path.parent.name,
                label=path.parent.parent.name
            ))

    logger.info("Retrieved %d results", len(results))
    return results


def print_documents(top_k_bm25,top_k=5):

    logging.info(f"-------------------Showing top {top_k} results for bm25-------------------")
    for document_k in top_k_bm25[:top_k]:

        rank = document_k["rank"]
        score = document_k["score"]
        grandparent = document_k["grandparent"]
        parent = document_k["parent"]

        logger.info(f"Rank {rank} (score: {score:.2f}) - {grandparent} - {parent}")