"""
skills/rag_query_engine.py  — Stage F1: RAG Query Engine
─────────────────────────────────────────────────────────
1. Embed the user's question  (text-embedding-3-small)
2. Cosine-similarity search against pgvector
3. Build a prompt with retrieved context
4. Call gpt-4o-mini and return the answer

Env vars read from .env:
  OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL
  PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_TABLE
  RAG_TOP_K, RAG_MAX_CONTEXT
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from skills.logger import StepLogger

log = StepLogger("rag_query")

CHAT_MODEL    = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o-mini")
EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DEFAULT_TABLE = os.getenv("PG_TABLE",           "data_embeddings")
TOP_K         = int(os.getenv("RAG_TOP_K",      "5"))
MAX_CONTEXT   = int(os.getenv("RAG_MAX_CONTEXT","4000"))


@dataclass
class RetrievedChunk:
    content:     str
    score:       float
    source:      str       = ""
    file_name:   str       = ""
    chunk_index: int       = 0
    metadata:    Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    question:          str
    answer:            str
    model:             str
    chunks_used:       int
    retrieved:         List[RetrievedChunk]
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    elapsed_s:         float = 0.0


_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def _get_pg_conn():
    conn = psycopg2.connect(
        host     = os.getenv("PG_HOST",     "localhost"),
        port     = int(os.getenv("PG_PORT", "5432")),
        dbname   = os.getenv("PG_DB",       "ragdb"),
        user     = os.getenv("PG_USER",     "postgres"),
        password = os.getenv("PG_PASSWORD", ""),
    )
    register_vector(conn)
    return conn


def _get_table_columns(conn, table: str) -> list[str]:
    """Return the column names for *table* so we can adapt the SELECT."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s ORDER BY ordinal_position;",
            (table,),
        )
        return [r[0] for r in cur.fetchall()]


def _get_table_vector_dim(table: str) -> Optional[int]:
    """Return the vector dimension stored in *table*, or None if undetectable."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5432")),
            dbname=os.getenv("PG_DB", "ragdb"),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", ""),
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT atttypmod FROM pg_attribute "
                "JOIN pg_class ON attrelid = pg_class.oid "
                "WHERE relname = %s AND attname = 'embedding';",
                (table,),
            )
            row = cur.fetchone()
        conn.close()
        return int(row[0]) if row and row[0] != -1 else None
    except Exception:
        return None


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _embed_question(question: str, dimensions: Optional[int] = None) -> List[float]:
    client = _get_client()
    kwargs: dict = {"model": EMBED_MODEL, "input": question}
    if dimensions and dimensions != 1536:
        kwargs["dimensions"] = dimensions
    resp = client.embeddings.create(**kwargs)
    return resp.data[0].embedding


def _retrieve_chunks(
    query_vec: List[float],
    table: str,
    top_k: int,
    filter_metadata: Optional[dict],
) -> List[RetrievedChunk]:
    conn = _get_pg_conn()
    try:
        cols = _get_table_columns(conn, table)

        # Map expected logical names to whatever columns exist in this table
        def pick(preferred: str, fallbacks: list[str]) -> Optional[str]:
            for name in [preferred] + fallbacks:
                if name in cols:
                    return name
            return None

        content_col     = pick("content",     ["text", "chunk_text", "document"])
        source_col      = pick("source",      ["file_path", "url"])
        file_name_col   = pick("file_name",   ["filename", "name", "source"])
        chunk_index_col = pick("chunk_index", ["chunk_id", "index"])
        embedding_col   = pick("embedding",   ["vector", "embeddings"])

        if not content_col or not embedding_col:
            raise ValueError(
                f"Table '{table}' must have a content column and an embedding column. "
                f"Found columns: {cols}"
            )

        select_parts = [f"{content_col} AS content"]
        if source_col:
            select_parts.append(f"{source_col} AS source")
        if file_name_col and file_name_col != source_col:
            select_parts.append(f"{file_name_col} AS file_name")
        if chunk_index_col:
            select_parts.append(f"{chunk_index_col} AS chunk_index")
        select_parts.append(
            f"1 - ({embedding_col} <=> %s::vector) AS score"
        )

        where_parts, filter_params = [], []
        if filter_metadata:
            for k, v in filter_metadata.items():
                if k in cols:
                    where_parts.append(f"{k} = %s")
                    filter_params.append(v)

        where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        sql = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM {table} {where_clause} "
            f"ORDER BY {embedding_col} <=> %s::vector "
            f"LIMIT %s;"
        )
        params = filter_params + [query_vec, query_vec, top_k]

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                content     = row.get("content") or "",
                score       = float(row.get("score") or 0),
                source      = row.get("source") or "",
                file_name   = row.get("file_name") or row.get("source") or "",
                chunk_index = row.get("chunk_index") or 0,
            )
            for row in rows
        ]
    finally:
        conn.close()


_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the user's
question using ONLY the context provided below. If the context does not
contain enough information, say so clearly. Always cite the source file
name when referencing specific information.

Context:
────────
{context}
────────"""


def _build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    parts, total = [], 0
    for c in chunks:
        label = c.file_name or c.source or "unknown"
        snippet = f"[{label} | chunk {c.chunk_index} | score {c.score:.3f}]\n{c.content}"
        if total + len(snippet) > MAX_CONTEXT:
            break
        parts.append(snippet)
        total += len(snippet)
    return _SYSTEM_PROMPT.format(context="\n\n".join(parts))


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(min=1, max=15),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _generate(system_prompt: str, question: str) -> tuple[str, int, int]:
    client = _get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        temperature=0.2,
    )
    usage = resp.usage
    return resp.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens


def answer_question(
    question:        str,
    table:           str = DEFAULT_TABLE,
    top_k:           int = TOP_K,
    filter_metadata: Optional[dict] = None,
) -> RAGResult:
    """
    Answer *question* using RAG over the pgvector table.

    Parameters
    ----------
    question        : natural language question
    table           : pgvector table (default: PG_TABLE env var)
    top_k           : number of chunks to retrieve (default 5)
    filter_metadata : optional column=value filters

    Returns
    -------
    RAGResult
    """
    t0 = time.perf_counter()
    log.start(f"Stage F1 - Query: '{question[:80]}'")

    vec_dim = _get_table_vector_dim(table)
    if vec_dim:
        log.step(f"Detected vector dimension: {vec_dim}")

    log.step("Embedding question ...")
    q_vec = _embed_question(question, dimensions=vec_dim)

    log.step(f"Retrieving top-{top_k} chunks from '{table}' ...")
    chunks = _retrieve_chunks(q_vec, table, top_k, filter_metadata)
    log.step(f"Retrieved {len(chunks)} chunks  [scores: {[round(c.score,3) for c in chunks]}]")

    log.step("Building prompt ...")
    system_prompt = _build_prompt(question, chunks)

    log.step(f"Calling {CHAT_MODEL} ...")
    answer, prompt_tok, compl_tok = _generate(system_prompt, question)
    elapsed = time.perf_counter() - t0

    log.done(
        f"F1 complete - {len(answer)} chars  "
        f"[tokens: {prompt_tok}+{compl_tok}, elapsed={elapsed:.1f}s]"
    )

    return RAGResult(
        question          = question,
        answer            = answer,
        model             = CHAT_MODEL,
        chunks_used       = len(chunks),
        retrieved         = chunks,
        prompt_tokens     = prompt_tok,
        completion_tokens = compl_tok,
        elapsed_s         = elapsed,
    )
