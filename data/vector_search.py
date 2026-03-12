"""Query interface for the ChromaDB vector store of NHTSA complaint narratives.

At query time, retrieves the most relevant real owner complaints for a
specific vehicle, optionally filtered by system and mileage range.

Computes query embeddings using the same SentenceTransformer model that
was used during the build phase (all-MiniLM-L6-v2).
"""

from __future__ import annotations

import logging
from pathlib import Path

from data.embed_complaints import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_collection = None
_embed_model = None


def _get_embed_model():
    """Lazy-load the SentenceTransformer model for query embedding."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    try:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        return _embed_model
    except Exception:
        logger.warning("Failed to load SentenceTransformer model", exc_info=True)
        return None


def _get_collection():
    """Lazy-load the ChromaDB collection (singleton)."""
    global _collection
    if _collection is not None:
        return _collection

    if not Path(CHROMA_DIR).exists():
        logger.warning("ChromaDB store not found at %s", CHROMA_DIR)
        return None

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
        logger.info(
            "Loaded ChromaDB collection '%s' with %d documents",
            COLLECTION_NAME,
            _collection.count(),
        )
        return _collection
    except Exception:
        logger.warning("Failed to load ChromaDB collection", exc_info=True)
        return None


def preload():
    """Pre-load ChromaDB + embedding model so first request isn't slow."""
    _get_collection()
    _get_embed_model()


def search_similar_complaints(
    make: str,
    model: str,
    year: int,
    system: str | None = None,
    mileage: int | None = None,
    n_results: int = 20,
) -> list[dict]:
    """Retrieve the most relevant complaint narratives from ChromaDB.

    Builds a semantic query string from the vehicle details and retrieves
    the top-k most similar complaint narratives. Results are filtered by
    make/model metadata, with optional year range and system filtering.

    Returns a list of dicts with keys: narrative, make, model, year,
    system, mileage, crash, fire, distance (similarity score).
    """
    collection = _get_collection()
    if collection is None:
        return []

    embed_model = _get_embed_model()
    if embed_model is None:
        return []

    query_parts = [f"{year} {make} {model}"]
    if system:
        query_parts.append(f"{system} problems")
    if mileage:
        query_parts.append(f"at {mileage} miles")
    query_text = " ".join(query_parts)

    query_embedding = embed_model.encode(
        [query_text], normalize_embeddings=True
    ).tolist()

    make_upper = make.strip().upper()
    model_upper = model.strip().upper()

    where_filter = {
        "$and": [
            {"make": {"$eq": make_upper}},
        ]
    }

    model_filter = {"model": {"$eq": model_upper}}
    where_filter["$and"].append(model_filter)

    if system:
        where_filter["$and"].append({"system": {"$eq": system}})

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, 50),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.debug(
            "Strict metadata filter returned no results, trying broader search"
        )
        try:
            broader_filter = {"make": {"$eq": make_upper}}
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, 50),
                where=broader_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            logger.warning("ChromaDB query failed", exc_info=True)
            return []

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append({
            "narrative": doc,
            "make": meta.get("make", ""),
            "model": meta.get("model", ""),
            "year": meta.get("year", 0),
            "system": meta.get("system", "Other"),
            "mileage": meta.get("mileage", 0),
            "crash": meta.get("crash", False),
            "fire": meta.get("fire", False),
            "distance": round(dist, 4),
        })

    if mileage:
        for item in output:
            item_mileage = item.get("mileage", 0)
            if item_mileage and item_mileage > 0:
                mileage_diff = abs(item_mileage - mileage)
                mileage_penalty = min(0.3, mileage_diff / 500_000)
                item["adjusted_distance"] = item["distance"] + mileage_penalty
            else:
                item["adjusted_distance"] = item["distance"] + 0.1
        output.sort(key=lambda x: x.get("adjusted_distance", x["distance"]))

    return output[:n_results]


def is_vector_store_available() -> bool:
    """Check if the vector store exists and has documents."""
    collection = _get_collection()
    if collection is None:
        return False
    try:
        return collection.count() > 0
    except Exception:
        return False
