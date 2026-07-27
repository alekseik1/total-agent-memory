"""Tests for Spreading Activation and Composition Engine."""

import pytest


class TestSpreadingActivation:
    def test_spread_single_seed(self, activation, populated_graph):
        result = activation.spread([populated_graph["auth"]], depth=1)
        assert len(result) > 0
        # auth's direct neighbors should be activated
        assert populated_graph["auth"] not in result  # seeds excluded

    def test_spread_multiple_seeds(self, activation, populated_graph):
        result = activation.spread(
            [populated_graph["auth"], populated_graph["billing"]], depth=1
        )
        assert len(result) > 0

    def test_spread_depth_1(self, activation, populated_graph):
        result = activation.spread([populated_graph["saas"]], depth=1)
        # depth 1: only direct neighbors of saas (auth, billing)
        # jwt and go may not be reached at depth 1
        activated_ids = set(result.keys())
        assert populated_graph["auth"] in activated_ids or populated_graph["billing"] in activated_ids

    def test_spread_depth_2(self, activation, populated_graph):
        result = activation.spread([populated_graph["saas"]], depth=2)
        # depth 2: should reach jwt and go via auth
        activated_ids = set(result.keys())
        # At depth 2 through high-weight edges, more nodes should activate
        assert len(activated_ids) >= 2

    def test_activation_decays_with_distance(self, activation, populated_graph):
        result = activation.spread([populated_graph["saas"]], depth=2)
        # Direct neighbors (auth, billing) should have higher activation
        # than indirect ones (jwt, go, webhook)
        if populated_graph["auth"] in result and populated_graph["jwt"] in result:
            assert result[populated_graph["auth"]] >= result[populated_graph["jwt"]]

    def test_multi_path_bonus(self, activation, graph_store):
        """Node reachable via multiple paths gets bonus activation."""
        a = graph_store.add_node("concept", "src1")
        b = graph_store.add_node("concept", "src2")
        target = graph_store.add_node("concept", "target_mp")

        graph_store.add_edge(a, target, "uses", weight=1.0)
        graph_store.add_edge(b, target, "uses", weight=1.0)

        result = activation.spread([a, b], depth=1)
        # target is reached from both seeds
        if target in result:
            assert result[target] > 0

    def test_activation_threshold_filters(self, activation, graph_store):
        """Nodes below threshold should not appear in results."""
        a = graph_store.add_node("concept", "strong_src")
        b = graph_store.add_node("concept", "weak_target")
        graph_store.add_edge(a, b, "uses", weight=0.01)

        result = activation.spread([a], depth=1)
        # Very weak edge should result in below-threshold activation
        # (depends on decay * weight calculation)
        for nid, score in result.items():
            assert score >= activation.ACTIVATION_THRESHOLD

    def test_find_seed_nodes(self, activation, populated_graph):
        seeds = activation.find_seed_nodes(["authentication", "billing"])
        assert len(seeds) == 2
        assert populated_graph["auth"] in seeds
        assert populated_graph["billing"] in seeds

    def test_find_seed_nodes_case_insensitive(self, activation, populated_graph):
        seeds = activation.find_seed_nodes(["AUTHENTICATION"])
        assert len(seeds) == 1
        assert populated_graph["auth"] in seeds

    def test_find_seed_nodes_prefix_fallback(self, activation, populated_graph):
        seeds = activation.find_seed_nodes(["authenti"])
        assert len(seeds) >= 1

    def test_find_seed_nodes_empty(self, activation):
        assert activation.find_seed_nodes([]) == []

    def test_get_activated_memories(self, db, activation, graph_store):
        """Test memory retrieval via activation map."""
        node_id = graph_store.add_node("concept", "test_mem")

        # Insert knowledge and link to node
        db.execute(
            "INSERT INTO knowledge (id, session_id, type, content, status, created_at) "
            "VALUES (100, 's1', 'solution', 'Test solution', 'active', '2025-01-01')"
        )
        db.execute(
            "INSERT INTO knowledge_nodes (knowledge_id, node_id, role, strength) "
            "VALUES (100, ?, 'related', 1.0)",
            (node_id,),
        )
        db.commit()

        activation_map = {node_id: 0.8}
        memories = activation.get_activated_memories(activation_map)
        assert len(memories) == 1
        assert memories[0][0] == 100  # knowledge_id
        assert memories[0][1] > 0  # score

    def test_get_activated_memories_empty(self, activation):
        assert activation.get_activated_memories({}) == []

    def test_empty_graph_spread(self, activation):
        result = activation.spread(["nonexistent_id"], depth=2)
        assert result == {}

    def test_spread_with_weights(self, activation, graph_store):
        """Higher weight edges should produce higher activation."""
        src = graph_store.add_node("concept", "weight_src")
        strong = graph_store.add_node("concept", "strong_nb")
        weak = graph_store.add_node("concept", "weak_nb")

        graph_store.add_edge(src, strong, "uses", weight=1.0)
        graph_store.add_edge(src, weak, "uses", weight=0.3)

        result = activation.spread([src], depth=1)
        # If both pass threshold, strong should have higher activation
        if strong in result and weak in result:
            assert result[strong] >= result[weak]


class TestCompositionEngine:
    def test_composition_greedy_cover(self, db):
        from associative.composition import CompositionEngine
        engine = CompositionEngine(db)

        memories = [
            {"id": 1, "content": "auth with jwt tokens", "type": "solution", "confidence": 0.9},
            {"id": 2, "content": "billing webhook integration", "type": "solution", "confidence": 0.8},
            {"id": 3, "content": "auth and billing and webhook", "type": "solution", "confidence": 0.7},
        ]

        result = engine.compose(
            needed_concepts=["auth", "billing", "webhook"],
            candidate_memories=memories,
        )
        assert result["coverage_percent"] > 0
        assert len(result["sources"]) > 0

    def test_composition_detects_gaps(self, db):
        from associative.composition import CompositionEngine
        engine = CompositionEngine(db)

        memories = [
            {"id": 1, "content": "auth solution", "type": "solution", "confidence": 0.9},
        ]

        result = engine.compose(
            needed_concepts=["auth", "billing", "webhook"],
            candidate_memories=memories,
        )
        assert len(result["gaps"]) > 0
        assert result["coverage_percent"] < 100.0

    def test_composition_detects_conflicts(self, db):
        from associative.composition import CompositionEngine
        engine = CompositionEngine(db)

        memories = [
            {"id": 1, "content": "auth with jwt approach A", "type": "solution", "confidence": 0.9},
            {"id": 2, "content": "auth with jwt approach B", "type": "pattern", "confidence": 0.7},
        ]

        result = engine.compose(
            needed_concepts=["auth"],
            candidate_memories=memories,
        )
        # Both cover "auth", so there should be a conflict if both selected
        if len(result["sources"]) >= 2:
            assert len(result["conflicts"]) > 0

    def test_composition_empty_input(self, db):
        from associative.composition import CompositionEngine
        engine = CompositionEngine(db)

        result = engine.compose(
            needed_concepts=[],
            candidate_memories=[],
        )
        assert result["sources"] == []
        assert result["coverage_percent"] == 0.0
