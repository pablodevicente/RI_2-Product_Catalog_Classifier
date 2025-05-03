def embed_query(
    query: str,
    model: Any,
    tokenize_fn: callable,
    idf_dict: Dict[str, float],
    vector_size: int
) -> np.ndarray:
    """
    Embeds a query into the same TF‑IDF-weighted vector space as the document corpus.

    Args:
        query (str): The raw query string.
        model: A pre-trained word embedding model or glove-index dict.
        tokenize_fn: A function that tokenizes raw text into a list of words.
        idf_dict: A mapping from terms to their inverse document frequency.
        vector_size: Dimensionality of the embedding vectors.

    Returns:
        np.ndarray: The query embedding vector.
    """
    tokens = tokenize_fn(query)

    vec = aux.tfidf_weighted_avg_embedding(
        doc_tokens=tokens,
        model=model,
        idf_dict=idf_dict,
        vector_size=vector_size
    )
    return vec
