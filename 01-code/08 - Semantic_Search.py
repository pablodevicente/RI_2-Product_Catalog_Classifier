import aux_document_retrieval_bm25 as aux_bm25
import aux_document_retrieval_vsm as aux_vsm
import aux_semantic_search as aux_semantics
import aux_document_retrieval_hybrid as aux_hybrid

from pathlib import Path
import logging
import nltk


nltk.download('punkt_tab')

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def main():
    setup_logging()

    paths = {
        'word2vec': Path("../02-data/03-VSM/01-Word2Vec/word2vec-google-news-300.bin"),
        'idf_cache': Path("../02-data/03-VSM/idf_cache_path.pkl"),
        'word2vec_vsm': Path("../02-data/03-VSM/01-Word2Vec/word2vec-4-50-4-150-0.pkl"),
        'file': Path("../02-data/00-testing/batteries-non-rechargable-primary/1cr2/1cr2.txt"),
        'output_path': Path("../02-data/00-testing/batteries-non-rechargable-primary/1cr2/sentence_expansions.txt"),
        'pdf_folder': Path("../02-data/00-testing/"),
        'retriever': Path("../02-data/05-Retrieval/corpus_bm25")
    }

    query = "this battery contains positive temperature coefficient element"
    top_k = 20

    try:
        top_k_vsm = aux_vsm.run_word2vec_query(paths, query, top_k=top_k)
        top_k_bm25 = aux_bm25.run_bm25_query(paths, query, top_k=top_k)

        aux_vsm.print_documents(top_k_vsm, top_k=top_k)
        aux_bm25.print_documents(top_k_bm25, top_k=top_k)

        top_k_hybrid = aux_hybrid.hybrid_retrieval(
            top_k_vsm,
            top_k_bm25,
            weight_vsm=0.7,
            weight_bm25=0.3,
            norm_vsm='minmax',
            norm_bm25='zscore'
        )
        aux_hybrid.print_documents(top_k_hybrid, top_k=top_k, ranking="hybrid")

        top_k_rrf = aux_hybrid.rrf(
            top_k_vsm["results"],
            top_k_bm25,
            k=60
        )
        aux_hybrid.print_documents(top_k_rrf, top_k=top_k, ranking="RRF")

        results = aux_hybrid.rerank(paths, query, top_k=top_k,mode="bm25-vsm")
        #aux_hybrid.rerank(paths, query, top_k=top_k,mode="vsm-bm25")

        for doc in results:
            print(
                f"{doc['rank']:2d} | id={doc['doc_id']:4d} "
                f"| bm25={doc['bm25_score']:.4f} "
                f"| vsm={doc['vsm_score']:.4f} "
                f"| path={doc['path'].name}"
            )


    except Exception as e:
        logging.error("Application error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()