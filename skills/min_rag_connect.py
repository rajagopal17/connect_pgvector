import os
import time
import psycopg2
from typing import List
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------
DB_CONFIG = {
    "host":     os.getenv("PG_HOST",     "localhost"),
    "port":     int(os.getenv("PG_PORT", "5432")),
    "dbname":   os.getenv("PG_DB",       "ragdb"),
    "user":     os.getenv("PG_USER",     "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

TABLE     = os.getenv("PG_TABLE", "data_embeddings")
TOP_K     = 5
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

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
def retrieve(query_vec: List[float]):
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT text, metadata_,
                   1 - (embedding <=> %s::vector) AS score
            FROM {TABLE}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vec, query_vec, TOP_K)
        )
        rows = cur.fetchall()

    conn.close()
    return rows

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
def main():
    question = "What different types of depreciation methods are mentioned in the document?"
    query_vec = get_embedding(question)
    chunks = retrieve(query_vec)
    prompt = build_prompt(chunks)
    #answer = generate(prompt, question)
    #print("Answer:", answer)
    #show the retrieved chunks and their scores for debugging
    for i, (text, metadata, score) in enumerate(chunks):
        print(f"Chunk {i+1} (score={score:.3f}):")
        print(f"Metadata: {metadata}")
        print(f"Text: {text}\n")

if __name__ == "__main__":
    main()