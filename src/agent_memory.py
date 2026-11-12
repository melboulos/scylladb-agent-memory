import json
import uuid
from datetime import datetime, timezone

from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement, ConsistencyLevel


class AgentMemory:
    def __init__(self, host="127.0.0.1"):
        self.cluster = Cluster([host], port=9042)
        self.session = self.cluster.connect("agent_platform")

        self.insert_stmt = self.session.prepare("""
            INSERT INTO agent_events (
                tenant_id, session_id, event_time,
                event_id, event_type, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
        """)

        self.read_stmt = self.session.prepare("""
            SELECT event_time, event_type, payload
            FROM agent_events
            WHERE tenant_id = ? AND session_id = ?
            LIMIT ?
        """)

    def write_event(self, tenant_id, session_id, event_type, payload):
        event_id = uuid.uuid4()
        event_time = datetime.now(timezone.utc)

        self.session.execute(
            self.insert_stmt,
            (
                tenant_id,
                session_id,
                event_time,
                event_id,
                event_type,
                json.dumps(payload),
            ),
        )
        return event_id

    def get_history(self, tenant_id, session_id, limit=20):
        rows = self.session.execute(
            self.read_stmt,
            (tenant_id, session_id, limit),
        )
        return [
            {
                "timestamp": row.event_time.isoformat(),
                "type": row.event_type,
                "payload": json.loads(row.payload),
            }
            for row in reversed(list(rows))
        ]

    def close(self):
        self.cluster.shutdown()


if __name__ == "__main__":
    memory = AgentMemory()

    try:
        tenant = "marriott"
        session_id = "demo-session-001"

        print("\nWriting AI agent events...\n")

        events = [
            (
                "customer_message",
                {"message": "Can I get a late checkout at 2 PM?"}
            ),
            (
                "agent_reasoning",
                {"action": "Check loyalty status and room availability"}
            ),
            (
                "tool_call",
                {"tool": "hotel_api", "operation": "check_availability"}
            ),
            (
                "tool_result",
                {"available": True, "checkout_time": "14:00"}
            ),
            (
                "agent_response",
                {"message": "Your late checkout is confirmed for 2 PM."}
            ),
        ]

        for event_type, payload in events:
            event_id = memory.write_event(
                tenant, session_id, event_type, payload
            )
            print(f"Stored {event_type}: {event_id}")

        print("\nReconstructed agent conversation:\n")

        for event in memory.get_history(tenant, session_id):
            print(f"[{event['type']}] {event['payload']}")

    finally:
        memory.close()
