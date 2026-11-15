import os
import ssl
from pathlib import Path

from dotenv import load_dotenv
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

hosts = os.environ["SCYLLA_HOSTS"].split(",")
port = int(os.getenv("SCYLLA_PORT", "19142"))
dc = os.environ["SCYLLA_DC"]
username = os.environ["SCYLLA_USERNAME"]
password = os.environ["SCYLLA_PASSWORD"]
ca_cert = ROOT / os.environ["SCYLLA_CA_CERT"]

profile = ExecutionProfile(
    load_balancing_policy=TokenAwarePolicy(
        DCAwareRoundRobinPolicy(local_dc=dc)
    )
)

ssl_context = ssl.create_default_context(cafile=str(ca_cert))

cluster = Cluster(
    contact_points=hosts,
    port=port,
    ssl_context=ssl_context,
    auth_provider=PlainTextAuthProvider(
        username=username,
        password=password
    ),
    execution_profiles={EXEC_PROFILE_DEFAULT: profile},
    connect_timeout=15,
)

try:
    print("Connecting to ScyllaDB Cloud...")
    session = cluster.connect()

    result = session.execute(
        "SELECT cluster_name, release_version FROM system.local"
    ).one()

    print("Connected successfully!")
    print("Cluster:", result.cluster_name)
    print("Version:", result.release_version)

    print("\nCluster nodes:")
    for host in cluster.metadata.all_hosts():
        print(f"  {host.address} ({host.datacenter})")

finally:
    cluster.shutdown()
