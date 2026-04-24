import os
import sys
import time
import psycopg2
from typing import List
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest # pyright: ignore[reportMissingImports]

load_dotenv()
sys.stdout.reconfigure(line_buffering=True)

# ----------------------------
# CONFIG
# ----------------------------
DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", "5434")),
    "dbname":   os.getenv("PG_DB"),
    "user":     os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}

TABLE        = os.getenv("PG_TABLE")
TOP_K        = int(os.getenv("TOP_K"))
EMBED_MODEL  = os.getenv("EMBED_MODEL")
CHAT_MODEL   = os.getenv("CHAT_MODEL")
RERANK_MODEL = os.getenv("RERANK_MODEL")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# EMBEDDING
# ----------------------------
def get_embedding(text: str) -> List[float]:
    dim = int(os.getenv("VECTOR_DIMENSION", "1536"))
    kwargs = {"input": text, "model": EMBED_MODEL}
    if dim != 1536:
        kwargs["dimensions"] = dim
    response = client.embeddings.create(**kwargs)
    return response.data[0].embedding
# ----------------------------
# RETRIEVE
# ----------------------------
def retrieve(query_vec):
    
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off;")
    cur.execute("SET hnsw.ef_search = 50;")
    cur.execute(f"""
        SELECT chunk_text, raw_metadata, embedding <=> %s::vector AS score
        FROM {TABLE}
        ORDER BY embedding <=> %s::vector
        LIMIT {TOP_K};
    """, (query_vec, query_vec))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# ----------------------------
# RERANK
# ----------------------------
def rerank(query: str, chunks: list) -> list:
    ranker = Ranker(model_name=RERANK_MODEL)
    passages = [{"id": i, "text": chunk[0], "meta": chunk[1]} for i, chunk in enumerate(chunks)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    return [(r["text"], r["meta"], r["score"]) for r in results]

# ----------------------------
# PROMPT
# ----------------------------
def build_prompt(chunks):
    context = "\n\n".join(
        f"[score={round(c[2],3)} | {c[1]}]\n{c[0]}" for c in chunks
    )

    return f"""
Answer using ONLY the context below.
If unsure, say you don't know.

Context:
{context}
"""

# ----------------------------
# GENERATE
# ----------------------------
def generate(prompt, question):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ----------------------------
# INDEX CHECK
# ----------------------------
def check_index(query_vec):
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off;")
    cur.execute("SET hnsw.ef_search = 50;")
    # List indexes on the table
    cur.execute(f"""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = '{TABLE}';
    """)
    indexes = cur.fetchall()
    print("Indexes on table:")
    for name, defn in indexes:
        print(f"  {name}: {defn}")
    # Show which index the planner uses for the vector query
    cur.execute(f"""
        EXPLAIN (FORMAT TEXT)
        SELECT chunk_text, raw_metadata, embedding <=> %s::vector AS score
        FROM {TABLE}
        ORDER BY embedding <=> %s::vector
        LIMIT {TOP_K};
    """, (query_vec, query_vec))
    plan = cur.fetchall()
    #print("\nQuery plan:")
    #for row in plan:
    #    print(" ", row[0])
    #cur.close()
    #conn.close()

# ----------------------------
def main():
    question = "how is risk management done in sap trm in not more than 5 lines?"
    query_vec = get_embedding(question)
    # check_index(query_vec)
    chunks = retrieve(query_vec)
    chunks = rerank(question, chunks)
    prompt = build_prompt(chunks)
    answer = generate(prompt, question)
    print("\n" + "="*60, flush=True)
    print("ANSWER", flush=True)
    print("="*60, flush=True)
    print(answer, flush=True)

    print("\n" + "="*60, flush=True)
    print("RERANKED CHUNKS", flush=True)
    print("="*60, flush=True)
    for i, (text, metadata, score) in enumerate(chunks):
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        page = metadata.get("page", "?") if isinstance(metadata, dict) else "?"
        source = metadata.get("source", "?").split("\\")[-1] if isinstance(metadata, dict) else "?"
        print(f"\n[{i+1}] score={score:.4f} | page={page} | {source}", flush=True)
        print("-"*50, flush=True)
        print(safe_text.strip(), flush=True)

if __name__ == "__main__":
    main()