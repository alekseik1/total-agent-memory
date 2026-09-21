"""
Graph Enricher — Enrich the knowledge graph with computed relationships.

Provides co-occurrence strengthening, PageRank computation,
community detection, transitive relation inference, and stale node pruning.

All operations work directly on SQLite via raw SQL for performance.
No external graph libraries required.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

LOG = lambda msg: sys.stderr.write(f"[memory-enricher] {msg}\n")


def _now() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class GraphEnricher:
    """Enrich the knowledge graph: co-occurrences, communities, PageRank."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db
        self.db.row_factory = sqlite3.Row

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def enrich_all(self) -> dict[str, Any]:
        """Run all enrichment steps. Returns stats dict."""
        LOG("Starting full enrichment...")

        cooccurrence_edges = self.strengthen_cooccurrences()
        pagerank_scores = self.compute_pagerank()
        communities = self.detect_communities()
        transitive = self.infer_transitive_relations()
        pruned = self.prune_stale_nodes()

        result = {
            "cooccurrence_edges": cooccurrence_edges,
            "pagerank_nodes_scored": len(pagerank_scores),
            "communities_found": len(communities),
            "transitive_edges_inferred": transitive,
            "stale_nodes_pruned": pruned,
        }

        LOG(f"Enrichment complete: {result}")
        return result

    def strengthen_cooccurrences(
        self,
        window_days: int = 30,
        min_count: int = 3,
    ) -> int:
        """Find concepts that appear together in knowledge records.

        Create or reinforce 'mentioned_with' edges between them.

        Algorithm:
        1. Get recent knowledge records (last N days)
        2. For each record, get linked concepts (via knowledge_nodes)
        3. For each pair of concepts in same record, increment co-occurrence count
        4. If count >= min_count, create/reinforce edge

        Returns number of edges created/reinforced.
        """
        # Query co-occurring node pairs in recent knowledge records
        rows = self.db.execute(
            """SELECT kn1.node_id AS node_a, kn2.node_id AS node_b, COUNT(*) AS cnt
               FROM knowledge_nodes kn1
               JOIN knowledge_nodes kn2
                 ON kn1.knowledge_id = kn2.knowledge_id
                 AND kn1.node_id < kn2.node_id
               JOIN knowledge k ON kn1.knowledge_id = k.id
               WHERE k.created_at >= strftime('%Y-%m-%dT%H:%M:%f000Z', 'now', ?)
                 AND k.status = 'active'
               GROUP BY kn1.node_id, kn2.node_id
               HAVING cnt >= ?
               ORDER BY cnt DESC""",
            (f"-{window_days} days", min_count),
        ).fetchall()

        edges_touched = 0
        now = _now()

        for row in rows:
            node_a, node_b, count = row["node_a"], row["node_b"], row["cnt"]

            # Weight scales with co-occurrence count: 0.5 base + 0.1 per occurrence (cap 5.0)
            weight = min(0.5 + count * 0.1, 5.0)

            # Try to update existing edge first
            updated = self.db.execute(
                """UPDATE graph_edges
                   SET weight = MAX(weight, ?),
                       last_reinforced_at = ?,
                       reinforcement_count = reinforcement_count + 1,
                       context = 'co-occurrence (' || ? || ' records)'
                   WHERE source_id = ? AND target_id = ? AND relation_type = 'mentioned_with'""",
                (weight, now, count, node_a, node_b),
            ).rowcount

            if not updated:
                # Verify both nodes exist and are active
                valid = self.db.execute(
                    """SELECT COUNT(*) FROM graph_nodes
                       WHERE id IN (?, ?) AND status = 'active'""",
                    (node_a, node_b),
                ).fetchone()[0]

                if valid == 2:
                    self.db.execute(
                        """INSERT INTO graph_edges
                           (id, source_id, target_id, relation_type, weight, context, created_at)
                           VALUES (?, ?, ?, 'mentioned_with', ?, ?, ?)""",
                        (uuid.uuid4().hex, node_a, node_b, weight,
                         f"co-occurrence ({count} records)", now),
                    )
                    edges_touched += 1
            else:
                edges_touched += 1

        self.db.commit()
        LOG(f"Strengthened {edges_touched} co-occurrence edges")
        return edges_touched

    def compute_pagerank(
        self,
        iterations: int = 20,
        damping: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank for all active nodes.

        Returns {node_id: score}.
        Also updates importance field on graph_nodes table.
        """
        # Load all active node IDs
        rows = self.db.execute(
            "SELECT id FROM graph_nodes WHERE status = 'active'"
        ).fetchall()
        node_ids = [r["id"] for r in rows]

        if not node_ids:
            LOG("No active nodes for PageRank")
            return {}

        n = len(node_ids)
        node_set = set(node_ids)

        # Build adjacency list: for each node, list of (neighbor, weight)
        outgoing: dict[str, list[tuple[str, float]]] = defaultdict(list)
        all_edges = self.db.execute(
            "SELECT source_id, target_id, weight FROM graph_edges"
        ).fetchall()

        for edge in all_edges:
            src, tgt, w = edge["source_id"], edge["target_id"], edge["weight"]
            if src in node_set and tgt in node_set:
                outgoing[src].append((tgt, w))
                outgoing[tgt].append((src, w))  # treat as undirected

        # Precompute outgoing weight sums for normalization
        out_weight_sum: dict[str, float] = {}
        for nid in node_ids:
            total = sum(w for _, w in outgoing.get(nid, []))
            out_weight_sum[nid] = total if total > 0 else 1.0

        # Initialize uniformly
        scores: dict[str, float] = {nid: 1.0 / n for nid in node_ids}
        base = (1.0 - damping) / n

        # Iterate
        for iteration in range(iterations):
            new_scores: dict[str, float] = {}
            for nid in node_ids:
                rank_sum = 0.0
                for neighbor, weight in outgoing.get(nid, []):
                    rank_sum += scores[neighbor] * (weight / out_weight_sum[neighbor])
                new_scores[nid] = base + damping * rank_sum
            scores = new_scores

        # Normalize to sum to 1.0
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}

        # Scale to [0, 1] relative to max and batch-update importance
        max_score = max(scores.values()) if scores else 1.0
        if max_score == 0:
            max_score = 1.0

        for node_id, score in scores.items():
            importance = round(score / max_score, 4)
            self.db.execute(
                "UPDATE graph_nodes SET importance = ? WHERE id = ?",
                (importance, node_id),
            )

        self.db.commit()
        LOG(f"Computed PageRank for {n} nodes ({iterations} iterations)")
        return scores

    def detect_communities(self, min_size: int = 3) -> list[list[str]]:
        """Simple community detection via connected components using BFS.

        Returns list of communities (each = list of node_ids),
        filtered by minimum size, sorted largest first.
        """
        # Load active nodes
        rows = self.db.execute(
            "SELECT id FROM graph_nodes WHERE status = 'active'"
        ).fetchall()
        node_ids = [r["id"] for r in rows]

        if not node_ids:
            return []

        node_set = set(node_ids)

        # Build adjacency list
        adjacency: dict[str, set[str]] = defaultdict(set)
        all_edges = self.db.execute(
            "SELECT source_id, target_id FROM graph_edges"
        ).fetchall()

        for edge in all_edges:
            src, tgt = edge["source_id"], edge["target_id"]
            if src in node_set and tgt in node_set:
                adjacency[src].add(tgt)
                adjacency[tgt].add(src)

        # BFS connected components
        visited: set[str] = set()
        communities: list[list[str]] = []

        for nid in node_ids:
            if nid in visited:
                continue

            # BFS from this node
            component: list[str] = []
            queue: deque[str] = deque([nid])
            visited.add(nid)

            while queue:
                current = queue.popleft()
                component.append(current)

                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) >= min_size:
                communities.append(component)

        # Sort by size descending
        communities.sort(key=len, reverse=True)
        LOG(f"Detected {len(communities)} communities (min_size={min_size})")
        return communities

    def infer_transitive_relations(self, max_new: int = 100) -> int:
        """Infer transitive relationships.

        Patterns:
        - A --uses--> B --depends_on--> C  =>  A --depends_on--> C (weight 0.3)
        - A --part_of--> B --part_of--> C  =>  A --part_of--> C (weight 0.3)
        - A --applies_to--> B --uses--> C  =>  A --applies_to--> C (weight 0.3)

        Only creates edges with low weight (0.3) and marks context as 'inferred'.
        Returns count of new edges created.
        """
        transitive_patterns: list[tuple[str, str, str]] = [
            # (rel1, rel2, inferred_rel)
            ("uses", "depends_on", "depends_on"),
            ("part_of", "part_of", "part_of"),
            ("applies_to", "uses", "applies_to"),
            ("provides", "requires", "depends_on"),
        ]

        created = 0
        now = _now()

        for rel1, rel2, inferred_rel in transitive_patterns:
            if created >= max_new:
                break

            # Find A->B (rel1) and B->C (rel2) where A->C doesn't exist
            candidates = self.db.execute(
                """SELECT e1.source_id AS a, e2.target_id AS c
                   FROM graph_edges e1
                   JOIN graph_edges e2 ON e1.target_id = e2.source_id
                   WHERE e1.relation_type = ?
                     AND e2.relation_type = ?
                     AND e1.source_id != e2.target_id
                     AND NOT EXISTS (
                         SELECT 1 FROM graph_edges e3
                         WHERE e3.source_id = e1.source_id
                           AND e3.target_id = e2.target_id
                           AND e3.relation_type = ?
                     )
                   LIMIT ?""",
                (rel1, rel2, inferred_rel, max_new - created),
            ).fetchall()

            for row in candidates:
                a_id, c_id = row["a"], row["c"]

                # Verify both nodes are active
                valid = self.db.execute(
                    """SELECT COUNT(*) FROM graph_nodes
                       WHERE id IN (?, ?) AND status = 'active'""",
                    (a_id, c_id),
                ).fetchone()[0]

                if valid != 2:
                    continue

                self.db.execute(
                    """INSERT OR IGNORE INTO graph_edges
                       (id, source_id, target_id, relation_type, weight, context, created_at)
                       VALUES (?, ?, ?, ?, 0.3, ?, ?)""",
                    (uuid.uuid4().hex, a_id, c_id, inferred_rel,
                     f"inferred: {rel1} + {rel2}", now),
                )
                created += 1

        self.db.commit()
        LOG(f"Inferred {created} transitive edges")
        return created

    def prune_stale_nodes(self, days: int = 180) -> int:
        """Archive nodes not seen in N days AND with no knowledge links.

        Sets status to 'archived' rather than deleting, preserving data.
        Returns count of archived nodes.
        """
        cursor = self.db.execute(
            """UPDATE graph_nodes
               SET status = 'archived'
               WHERE status = 'active'
                 AND last_seen_at < datetime('now', ?)
                 AND id NOT IN (
                     SELECT DISTINCT node_id FROM knowledge_nodes
                 )
                 AND mention_count <= 1""",
            (f"-{days} days",),
        )
        self.db.commit()

        count = cursor.rowcount
        if count > 0:
            LOG(f"Archived {count} stale nodes (not seen in {days} days)")
        return count

    def stats(self) -> dict[str, Any]:
        """Graph health statistics.

        Returns:
        {
            "total_nodes": int,
            "total_edges": int,
            "nodes_by_type": {...},
            "edges_by_type": {...},
            "avg_connectivity": float,
            "orphan_nodes": int,
            "top_nodes": [...],  # by importance (top 10)
            "communities": int,
        }
        """
        # Total counts
        total_nodes = self.db.execute(
            "SELECT COUNT(*) FROM graph_nodes WHERE status = 'active'"
        ).fetchone()[0]

        total_edges = self.db.execute(
            "SELECT COUNT(*) FROM graph_edges"
        ).fetchone()[0]

        # Nodes by type
        nodes_by_type: dict[str, int] = {}
        for row in self.db.execute(
            "SELECT type, COUNT(*) AS cnt FROM graph_nodes WHERE status = 'active' GROUP BY type"
        ).fetchall():
            nodes_by_type[row["type"]] = row["cnt"]

        # Edges by type
        edges_by_type: dict[str, int] = {}
        for row in self.db.execute(
            "SELECT relation_type, COUNT(*) AS cnt FROM graph_edges GROUP BY relation_type"
        ).fetchall():
            edges_by_type[row["relation_type"]] = row["cnt"]

        # Average connectivity (edges per node)
        avg_connectivity = round(total_edges / total_nodes, 2) if total_nodes > 0 else 0.0

        # Orphan nodes (no edges, no knowledge links)
        orphan_nodes = self.db.execute(
            """SELECT COUNT(*) FROM graph_nodes
               WHERE status = 'active'
                 AND id NOT IN (
                     SELECT DISTINCT source_id FROM graph_edges
                     UNION
                     SELECT DISTINCT target_id FROM graph_edges
                 )
                 AND id NOT IN (
                     SELECT DISTINCT node_id FROM knowledge_nodes
                 )"""
        ).fetchone()[0]

        # Top nodes by importance
        top_rows = self.db.execute(
            """SELECT name, type, importance, mention_count
               FROM graph_nodes
               WHERE status = 'active'
               ORDER BY importance DESC
               LIMIT 10"""
        ).fetchall()
        top_nodes = [
            {
                "name": r["name"],
                "type": r["type"],
                "importance": r["importance"],
                "mention_count": r["mention_count"],
            }
            for r in top_rows
        ]

        # Community count
        communities = self.detect_communities(min_size=3)

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "avg_connectivity": avg_connectivity,
            "orphan_nodes": orphan_nodes,
            "top_nodes": top_nodes,
            "communities": len(communities),
        }


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main() -> None:
    """Run enricher from command line."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Enrich the knowledge graph")
    parser.add_argument("--db", default=str(Path.home() / ".claude-memory" / "memory.db"),
                        help="Path to SQLite database")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only show stats, don't enrich")
    parser.add_argument("--cooccurrences", action="store_true",
                        help="Only run co-occurrence strengthening")
    parser.add_argument("--pagerank", action="store_true",
                        help="Only run PageRank")
    parser.add_argument("--communities", action="store_true",
                        help="Only detect communities")
    parser.add_argument("--prune", action="store_true",
                        help="Only prune stale nodes")
    args = parser.parse_args()

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")

    enricher = GraphEnricher(db)

    if args.stats_only:
        result = enricher.stats()
    elif args.cooccurrences:
        result = {"cooccurrence_edges": enricher.strengthen_cooccurrences()}
    elif args.pagerank:
        scores = enricher.compute_pagerank()
        result = {"nodes_scored": len(scores)}
    elif args.communities:
        communities = enricher.detect_communities()
        result = {"communities": len(communities), "sizes": [len(c) for c in communities]}
    elif args.prune:
        result = {"pruned": enricher.prune_stale_nodes()}
    else:
        result = enricher.enrich_all()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
