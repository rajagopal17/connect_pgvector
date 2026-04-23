"""
query.py — Simple RAG query runner
────────────────────────────────────
Connects to the pgvector table defined in .env, embeds the question with
OpenAI, retrieves top-5 chunks, and answers with gpt-4o-mini.

Usage:
    python query.py                         # runs built-in example question
    python query.py "your question here"    # custom question
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from skills.rag_query_engine import answer_question

QUESTION = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "What is the main topic covered in the stored documents?"
)

result = answer_question(question=QUESTION, top_k=5)

print("\n" + "=" * 70)
print(f"Question : {result.question}")
print("=" * 70)
print(f"\n{result.answer}\n")
print("-" * 70)
print(f"Model    : {result.model}")
print(f"Chunks   : {result.chunks_used}")
print(f"Tokens   : {result.prompt_tokens} prompt / {result.completion_tokens} completion")
print(f"Elapsed  : {result.elapsed_s:.1f}s")
print("-" * 70)
print("\nRetrieved chunks:")
for i, chunk in enumerate(result.retrieved, 1):
    label = chunk.file_name or chunk.source or "unknown"
    print(f"  [{i}] score={chunk.score:.4f}  source={label}  chunk={chunk.chunk_index}")
    preview = chunk.content[:120].strip().encode("ascii", errors="replace").decode("ascii")
    print(f"       {preview}...")
