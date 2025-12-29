from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set, Optional

import torch
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker


# =========================
# CONFIG
# =========================
@dataclass
class RAGConfig:
    # Supabase
    supabase_url: str
    supabase_key: str
    rpc_function: str = "match_law_chunks"

    device: str = "cuda"

    # Models
    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Retrieval params
    match_threshold: float = 0.0
    top_k_retrieval: int = 30
    top_k_final: int = 5

    # Context params
    max_context_chars: int = 7000
    
    only_send_top1_to_llm: bool = True
    show_sources: bool = True


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class SearchResult:
    source_id: str
    score: float
    small_chunk: str
    full_context: str
    id_path: str


# =========================
# MODEL MANAGER
# =========================
class ModelManager:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

        # 1) Load Embedding Model (E5)
        self.embedder = SentenceTransformer(cfg.embedding_model, device=cfg.device)

        # 2) Load Reranker Model (BGE)
        self.reranker = FlagReranker(cfg.reranker_model, use_fp16=(cfg.device == "cuda"))

        # 3) Connect Supabase
        if not cfg.supabase_url or not cfg.supabase_key:
            raise ValueError("Thiếu Supabase Credentials!")
        self.db_client: Client = create_client(cfg.supabase_url, cfg.supabase_key)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.encode(f"query: {text}", normalize_embeddings=True).tolist()

    def rerank_pairs(self, pairs: List[List[str]]) -> List[float]:
        if not pairs:
            return []
        scores = self.reranker.compute_score(pairs)
        return scores if isinstance(scores, list) else [scores]


# =========================
# RETRIEVER
# =========================
class AdvancedRetriever:
    def __init__(self, mm: ModelManager, cfg: RAGConfig):
        self.mm = mm
        self.cfg = cfg

    def run(self, query: str) -> List[SearchResult]:
        # STEP 1: RETRIEVAL (RECALL WIDE)
        query_vec = self.mm.embed_query(query)

        rpc_params = {
            "query_embedding": query_vec,
            "match_threshold": self.cfg.match_threshold,
            "match_count": self.cfg.top_k_retrieval,
        }

        try:
            response = self.mm.db_client.rpc(self.cfg.rpc_function, rpc_params).execute()
            candidates = response.data or []
        except Exception:
            return []

        if not candidates:
            return []

        # STEP 2: RERANKING (PRECISION)
        pairs = [[query, item.get("content", "")] for item in candidates]
        scores = self.mm.rerank_pairs(pairs)

        for i, item in enumerate(candidates):
            item["rerank_score"] = scores[i]

        ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # STEP 3: EXPANSION & DEDUPLICATION
        final_results: List[SearchResult] = []
        seen_parent_ids: Set[str] = set()

        for item in ranked_candidates:
            if len(final_results) >= self.cfg.top_k_final:
                break

            meta = item.get("metadata", {}) or {}
            if not isinstance(meta, dict):
                meta = {}

            parent_id = meta.get("article")

            # fallback nhỏ để tránh parent_id=None làm dedup sai
            if not isinstance(parent_id, str) or not parent_id.strip():
                parent_id = self._parent_id_from_id_path(meta.get("id_path")) or "UNKNOWN_PARENT"

            # dedup theo parent
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)

            content = item.get("content", "") or ""
            full_context = meta.get("full_context", content) or content

            final_results.append(
                SearchResult(
                    source_id=parent_id,
                    score=float(item["rerank_score"]),
                    small_chunk=content,
                    full_context=full_context,
                    id_path=str(meta.get("id_path", "N/A")),
                )
            )

        return final_results

    @staticmethod
    def _parent_id_from_id_path(id_path: Any) -> str:
        """
        id_path ví dụ: ChI|M1|D15|K1 -> parent: ChI|M1|D15
        """
        if not isinstance(id_path, str):
            return ""
        parts = [p for p in id_path.split("|") if p]
        parent_parts: []
        for p in parts:
            parent_parts.append(p)
            if p.startswith("D"):
                break
        return "|".join(parent_parts)


# =========================
# PUBLIC API
# =========================
_MM: Optional[ModelManager] = None
_RETRIEVER: Optional[AdvancedRetriever] = None


def init_rag(cfg: RAGConfig) -> Tuple[ModelManager, AdvancedRetriever]:
    global _MM, _RETRIEVER

    if _MM is None:
        try:
            torch.set_num_threads(2)
        except Exception:
            pass
        _MM = ModelManager(cfg)

    if _RETRIEVER is None:
        _RETRIEVER = AdvancedRetriever(_MM, cfg)

    return _MM, _RETRIEVER

def create_supabase_client(cfg: RAGConfig) -> Client:
    return create_client(cfg.supabase_url, cfg.supabase_key)

def retrieve_context(
    supabase: Client,
    cfg: RAGConfig,
    query: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Trả về:
      - context (string): ghép full_context của các parent sau dedup
      - sources (list): dùng cho UI expander (nếu cfg.show_sources=True)
    """
    _, retriever = init_rag(cfg)
    results = retriever.run(query)

    if not results:
        return "", []

    if cfg.only_send_top1_to_llm:
        results = results[:1]

    parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for i, r in enumerate(results, start=1):
        ctx = (r.full_context or "").strip()
        if not ctx:
            continue
        parts.append(f"[{i}] {r.source_id}\n{ctx}")

        if cfg.show_sources:
            sources.append(
                {
                    "source": r.source_id,
                    "id_path": r.id_path,
                    "article": r.source_id,
                    "score": r.score,
                }
            )

    context = "\n\n".join(parts)[: cfg.max_context_chars]

    return context, sources
