import os
import ssl
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

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

    # New, isolated test keyspace with tablets enabled.
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS ai_memory_lab
        WITH replication = {
            'class': 'NetworkTopologyStrategy',
            'AWS_US_EAST_1': 3
        }
        AND tablets = {'enabled': true}
    """)

    session.execute("""
        CREATE TABLE IF NOT EXISTS ai_memory_lab.vector_test (
            id uuid PRIMARY KEY,
            content text,
            embedding vector<float, 3>
        )
    """)

    print("Creating native vector index...")
    session.execute("""
        CREATE CUSTOM INDEX IF NOT EXISTS vector_test_ann_idx
        ON ai_memory_lab.vector_test(embedding)
        USING 'vector_index'
        WITH OPTIONS = {'similarity_function': 'COSINE'}
    """)

    examples = [
        ("ScyllaDB stores AI agent conversation history",
         [0.9, 0.1, 0.0]),
        ("Vector databases support semantic memory retrieval",
         [0.8, 0.2, 0.1]),
        ("The hotel has a swimming pool",
         [0.0, 0.1, 0.9]),
    ]

    for content, vector in examples:
        values = ", ".join(str(v) for v in vector)
        escaped = content.replace("'", "''")
        session.execute(f"""
            INSERT INTO ai_memory_lab.vector_test
            (id, content, embedding)
            VALUES ({uuid.uuid4()}, '{escaped}', [{values}])
        """)

    print("Inserted 3 test vectors")

    query = """
        SELECT content,
               similarity_cosine(
                   embedding, [0.95, 0.15, 0.05]
               ) AS score
        FROM ai_memory_lab.vector_test
        ORDER BY embedding ANN OF [0.95, 0.15, 0.05]
        LIMIT 3
    """

    # Index propagation is asynchronous.
    for attempt in range(12):
        try:
            start = time.perf_counter()
            rows = list(session.execute(query))
            latency = (time.perf_counter() - start) * 1000

            if len(rows) == 3:
                print(f"\nANN query latency: {latency:.2f} ms")
                for row in rows:
                    print(f"{row.score:.4f} | {row.content}")
                break

            print(f"Indexing: {len(rows)}/3 results visible")
        except Exception as exc:
            print(f"Index not ready: {str(exc)[:180]}")

        time.sleep(5)
    else:
        print("Index not fully ready. Retry in a minute.")

finally:
    cluster.shutdown()
