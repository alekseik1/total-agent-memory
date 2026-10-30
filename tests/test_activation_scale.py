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


def test_spread_skips_hub_nodes(activation, graph_store, monkeypatch):
    monkeypatch.setenv("MEMORY_GRAPH_HUB_DEGREE", "5")
    seed = graph_store.add_node("concept", "seed")
    typed_hub = graph_store.add_node("project", "big-project")
    busy = graph_store.add_node("concept", "busy")
    behind_typed = graph_store.add_node("concept", "behind-typed")
    behind_busy = graph_store.add_node("concept", "behind-busy")
    graph_store.add_edge(seed, typed_hub, "relates", weight=0.9)
    graph_store.add_edge(seed, busy, "relates", weight=0.9)
    graph_store.add_edge(typed_hub, behind_typed, "relates", weight=0.9)
    graph_store.add_edge(busy, behind_busy, "relates", weight=0.9)
    for i in range(10):
        graph_store.add_edge(busy, graph_store.add_node("concept", f"filler{i}"), "relates", weight=0.9)

    activated = activation.spread([seed], depth=2)

    assert typed_hub not in activated
    assert behind_typed not in activated
    assert behind_busy not in activated


def test_activated_memories_accept_more_nodes_than_sqlite_binds(db, activation):
    insert_knowledge(db, 1)
    node = "n0"
    db.execute("INSERT INTO knowledge_nodes (knowledge_id, node_id, role, strength) VALUES (1, ?, 'mentions', 1.0)", (node,))
    db.commit()
    huge = {f"n{i}": 0.5 for i in range(20000)}
    huge[node] = 0.9

    assert activation.get_activated_memories(huge, top_k=5) == [(1, 0.9)]
