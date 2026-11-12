CREATE KEYSPACE IF NOT EXISTS agent_platform
WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'replication_factor': 1
};

CREATE TABLE IF NOT EXISTS agent_platform.agent_events (
    tenant_id text,
    session_id text,
    event_time timestamp,
    event_id uuid,
    event_type text,
    payload text,
    PRIMARY KEY (
        (tenant_id, session_id),
        event_time,
        event_id
    )
) WITH CLUSTERING ORDER BY (
    event_time DESC,
    event_id ASC
);
