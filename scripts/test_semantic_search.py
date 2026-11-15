import os
import ssl
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

print("Loading local embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

profile = ExecutionProfile(
    load_balancing_policy=TokenAwarePolicy(
        DCAwareRoundRobinPolicy(local_dc=os.environ["SCYLLA_DC"])
    )
)

cluster = Cluster(
    contact_points=os.environ["SCYLLA_HOSTS"].split(","),
    port=int(os.getenv("SCYLLA_PORT", "19142")),
    ssl_context=ssl.create_default_context(
        cafile=str(ROOT / os.environ["SCYLLA_CA_CERT"])
    ),
    auth_provider=PlainTextAuthProvider(
        username=os.environ["SCYLLA_USERNAME"],
        password=os.environ["SCYLLA_PASSWORD"]
    ),
    execution_profiles={EXEC_PROFILE_DEFAULT: profile},
)

try:
    session = cluster.connect()
    print("Connected to ScyllaDB Cloud")

    session.execute("""
        CREATE TABLE IF NOT EXISTS ai_memory_lab.semantic_memories (
            id uuid PRIMARY KEY,
            content text,
            embedding vector<float, 384>
        )
    """)

    print("Creating native 384-dimensional vector index...")
    session.execute("""
        CREATE CUSTOM INDEX IF NOT EXISTS semantic_memories_ann_idx
        ON ai_memory_lab.semantic_memories(embedding)
        USING 'vector_index'
        WITH OPTIONS = {'similarity_function': 'COSINE'}
    """)

    memories = [
        "I spent twelve years working with distributed databases.",
        "I enjoy designing enterprise AI agent architectures.",
        "ScyllaDB uses a shard-per-core architecture.",
        "My favorite vacation destination is the beach.",
        "Vector search retrieves information by semantic similarity.",
        "I have extensive experience with enterprise technical sales."
    ]

    print("Generating and inserting real embeddings...")
    vectors = model.encode(memories, normalize_embeddings=True)

    insert = session.prepare("""
        INSERT INTO ai_memory_lab.semantic_memories
        (id, content, embedding)
        VALUES (?, ?, ?)
    """)

    for content, vector in zip(memories, vectors):
        session.execute(insert, (uuid.uuid4(), content, vector.tolist()))

    print(f"Inserted {len(memories)} semantic memories")

    question = "What is my background in database technology?"
    query_vector = model.encode(
        question, normalize_embeddings=True
    ).tolist()

    vector_literal = "[" + ", ".join(map(str, query_vector)) + "]"

    query = f"""
        SELECT content,
               similarity_cosine(
                   embedding, {vector_literal}
               ) AS score
        FROM ai_memory_lab.semantic_memories
        ORDER BY embedding ANN OF {vector_literal}
        LIMIT 5
    """

    print(f"\nQuestion: {question}")

    for attempt in range(12):
        try:
            start = time.perf_counter()
            rows = list(session.execute(query))
            latency = (time.perf_counter() - start) * 1000

            if len(rows) >= len(memories) - 1:
                print(f"ScyllaDB ANN latency: {latency:.2f} ms\n")
                for row in rows:
                    print(f"{row.score:.4f} | {row.content}")
                break

            print(f"Waiting for index: {len(rows)}/5 results")
        except Exception as exc:
            print(f"Index not ready: {str(exc)[:180]}")

        time.sleep(5)
    else:
        print("Index still catching up; rerun the test shortly.")

finally:
    cluster.shutdown()
