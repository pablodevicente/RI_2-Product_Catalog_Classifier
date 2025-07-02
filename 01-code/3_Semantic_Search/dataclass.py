from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

## classes for both BM25 and VSM managing

@dataclass
class ChunkEntry:
    doc_id: str
    full_path: str
    vector: np.ndarray
    parent: str
    grandparent: str
    score: float = field(default=None)
    idx: int = field(default=None)

@dataclass
class QueryEmbeddingResult:
    query: str
    vector: np.ndarray
    tokens: List[str]
    expansions: List[str]

@dataclass
class RetrievedDocument:
    """Metadatos y puntuación de un documento recuperado."""
    rank: int
    doc_id: str
    label: str
    score: float

@dataclass
class TopKDocumentsResult:
    """Resultado de la recuperación top-k."""
    top_k: int
    documents: List[RetrievedDocument]
    multi_vector: Optional[bool] = None

## I know they are reduntant, dont care.

@dataclass
class Word2VecQueryResult:
    """
    Resultado completo de una consulta Word2Vec pre-cargada.
    - query_info: información detallada de la query y embeddings.
    - results: documentos recuperados con sus metadatos y puntuaciones.
    """
    query_info: QueryEmbeddingResult
    results: TopKDocumentsResult

@dataclass
class BM25QueryResult:
    """
    - query_info: información detallada de la query
    - results: documentos recuperados con sus metadatos y puntuaciones.
    """
    query_info: str
    results: TopKDocumentsResult

@dataclass
class QueryResult:
    """
    - results: documentos recuperados con sus metadatos y puntuaciones.
    """
    results: TopKDocumentsResult

## for managing multiple queries
@dataclass
class DocumentSection:
    doc: str
    label: str
    query: str