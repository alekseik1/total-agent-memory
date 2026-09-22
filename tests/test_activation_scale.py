"""Spreading activation reads only what it touches."""

from __future__ import annotations


def test_spread_reads_only_frontier_edges(db, activation, graph_store, populated_graph):
    far = [graph_store.add_node("concept", f"unrelated{i}") for i in range(3)]
    graph_store.add_edge(far[0], far[1], "relates", weight=0.9)
    statements = []
    db.set_trace_callback(statements.append)
    activated = activation.spread([populated_graph["saas"]], depth=2)
    db.set_trace_callback(None)

    assert populated_graph["auth"] in activated
    assert not set(far) & set(activated)
    edge_reads = [s for s in statements if "FROM graph_edges" in s]
    assert edge_reads and all("IN (" in s for s in edge_reads)


def insert_knowledge(db, kid, status="active"):
    db.execute(
        "INSERT INTO knowledge (id, session_id, type, content, status, created_at) "
        "VALUES (?, 's1', 'fact', ?, ?, '2026-01-01T00:00:00Z')",
        (kid, f"k{kid}", status),
    )


def test_activated_memories_sum_in_sql(db, activation, graph_store):
    hub = graph_store.add_node("concept", "hub")
    leaf = graph_store.add_node("concept", "leaf")
    for kid in (1, 2, 3):
        insert_knowledge(db, kid)
    insert_knowledge(db, 4, status="superseded")
    links = [(1, hub, 0.5), (1, leaf, None), (2, hub, 0.0), (3, leaf, 2.0), (4, hub, 1.0)]
    for kid, node, strength in links:
        db.execute("INSERT INTO knowledge_nodes (knowledge_id, node_id, role, strength) VALUES (?, ?, 'mentions', ?)",
                   (kid, node, strength))
    db.commit()

    ranked = activation.get_activated_memories({hub: 0.8, leaf: 0.5}, top_k=10)

    # 1: 0.8*0.5 + 0.5*1.0 (NULL -> 1.0); 2: 0.8*1.0 (0 -> 1.0); 3: 0.5*2.0; 4 is not active.
    assert ranked == [(3, 1.0), (1, 0.9), (2, 0.8)]
    assert activation.get_activated_memories({hub: 0.8, leaf: 0.5}, top_k=1) == [(3, 1.0)]
