"""Embed NHTSA complaint narratives into a ChromaDB vector store for RAG.

Uses sentence-transformers (all-MiniLM-L6-v2) to create embeddings of
complaint narratives, stored in a persistent ChromaDB collection on disk.
At query time, the vector store lets us retrieve the most relevant real
owner complaints for any specific vehicle + issue combination.

Optimizations:
- CUDA GPU acceleration when available (RTX 5060, etc.)
- Resume support: continues from where it left off if interrupted
- Deduplication: skips complaints with identical narrative text
- Pre-computes embeddings in large GPU batches before inserting into ChromaDB
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from config.settings import BASE_DIR
from data.bulk_loader import NHTSAComplaint, _get_bulk_engine, BulkBase

logger = logging.getLogger(__name__)

CHROMA_DIR = BASE_DIR / "data" / "chroma_store"
COLLECTION_NAME = "nhtsa_complaints"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_BATCH_SIZE = 10_000
MAX_NARRATIVE_LEN = 512


def _get_chroma_client():
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_embedding_fn():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA GPU detected: %s", name)
            return "cuda"
    except Exception:
        pass
    logger.info("No CUDA GPU found, using CPU")
    return "cpu"


def _create_gpu_model(device: str):
    """Create a SentenceTransformer model on the specified device."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    logger.info("Loaded embedding model '%s' on %s", EMBEDDING_MODEL, device)
    return model


def _get_last_embedded_id(collection) -> int:
    """Find the highest DB row ID already in the collection to enable resume."""
    try:
        count = collection.count()
        if count == 0:
            return 0

        results = collection.get(
            limit=1,
            offset=count - 1,
            include=[],
        )
        if results and results["ids"]:
            last_id_str = results["ids"][0]
            parts = last_id_str.split("_")
            if len(parts) >= 2:
                return int(parts[-1])
    except Exception:
        logger.debug("Could not determine last embedded ID", exc_info=True)
    return 0


def build_vector_store(reset: bool = False):
    """Embed complaint narratives from nhtsa_bulk.db into ChromaDB.

    Uses CUDA GPU when available for ~10x faster embedding.
    Supports resuming from where it left off. Set reset=True to start fresh.
    Deduplicates narratives to reduce embedding work.
    """
    device = _detect_device()
    gpu_batch_size = 512 if device == "cuda" else 128

    engine = _get_bulk_engine()
    BulkBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    total_rows = (
        session.query(func.count(NHTSAComplaint.id))
        .filter(NHTSAComplaint.narrative.isnot(None))
        .filter(NHTSAComplaint.narrative != "")
        .filter(func.length(NHTSAComplaint.narrative) > 20)
        .scalar()
    )
    logger.info("Total complaints with narratives: %d", total_rows)

    model = _create_gpu_model(device)

    client = _get_chroma_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    resume_after_id = 0

    if existing_count > 0 and not reset:
        resume_after_id = _get_last_embedded_id(collection)
        logger.info(
            "Resuming: %d documents already in collection, continuing after DB row ID %d",
            existing_count, resume_after_id,
        )

    embedded_count = existing_count
    seen_hashes: set[str] = set()
    skipped_dupes = 0

    while True:
        rows = (
            session.query(NHTSAComplaint)
            .filter(NHTSAComplaint.narrative.isnot(None))
            .filter(NHTSAComplaint.narrative != "")
            .filter(func.length(NHTSAComplaint.narrative) > 20)
            .filter(NHTSAComplaint.id > resume_after_id)
            .order_by(NHTSAComplaint.id)
            .limit(DB_BATCH_SIZE)
            .all()
        )

        if not rows:
            break

        resume_after_id = rows[-1].id

        batch_ids = []
        batch_docs = []
        batch_metas = []

        for row in rows:
            narrative = row.narrative[:MAX_NARRATIVE_LEN].strip()
            if not narrative:
                continue

            text_hash = hashlib.md5(narrative.encode("utf-8", errors="replace")).hexdigest()
            if text_hash in seen_hashes:
                skipped_dupes += 1
                continue
            seen_hashes.add(text_hash)

            batch_ids.append(f"{row.cmpl_id}_{row.id}")
            batch_docs.append(narrative)
            batch_metas.append({
                "make": row.make or "",
                "model": row.model or "",
                "year": row.year or 0,
                "system": row.system or "Other",
                "mileage": row.mileage or 0,
                "crash": row.crash or False,
                "fire": row.fire or False,
            })

        if not batch_docs:
            continue

        embeddings = model.encode(
            batch_docs,
            batch_size=gpu_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        chroma_chunk = 5000
        for i in range(0, len(batch_ids), chroma_chunk):
            end = min(i + chroma_chunk, len(batch_ids))
            collection.add(
                ids=batch_ids[i:end],
                documents=batch_docs[i:end],
                metadatas=batch_metas[i:end],
                embeddings=embeddings[i:end],
            )

        embedded_count += len(batch_ids)

        if embedded_count % 25_000 < DB_BATCH_SIZE or not rows:
            logger.info(
                "  ... embedded %d total (%d new this run, %d dupes skipped) [%s]",
                embedded_count,
                embedded_count - existing_count,
                skipped_dupes,
                device.upper(),
            )

    session.close()
    logger.info(
        "Vector store build complete: %d total documents in '%s' (%d dupes skipped). Store at %s",
        embedded_count, COLLECTION_NAME, skipped_dupes, CHROMA_DIR,
    )
    return embedded_count
