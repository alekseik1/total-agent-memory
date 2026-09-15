from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKER_TIMEOUT_SECONDS = 120


def worker(connection, code_root, kind, record_usage):
    with tempfile.TemporaryDirectory() as directory:
        with (
            sqlite3.connect(f'file:{ROOT}/backups/standard-v14-20260911/{kind}.db?mode=ro', uri=True) as source,
            sqlite3.connect(Path(directory) / 'memory.db') as target,
        ):
            source.backup(target)
        os.environ.update(TAM_MEMORY_DIR=directory, CLAUDE_MEMORY_DIR=directory,
                          MEMORY_MODE='fast', MEMORY_LLM_ENABLED='false', MEMORY_ASYNC_ENRICHMENT='false',
                          MEMORY_OUTBOX_ENABLED='false', MEMORY_QUALITY_GATE_ENABLED='false')
        sys.path.insert(0, str(code_root / 'src'))
        import server
        from memory_core.retrieval import flatten_results

        store = server.Store()
        recall = server.Recall(store)
        try:
            connection.send({'ready': True})
            while request := connection.recv():
                if store.cache is not None:
                    store.cache.invalidate()
                if getattr(store, 'v9_cache', None) is not None:
                    store.v9_cache.invalidate_all()
                if request['temperature'] == 'cold' and hasattr(store, '_vector_search'):
                    store._vector_search.clear()
                project = store.db.execute('SELECT project FROM knowledge WHERE id=?', (request['anchor'],)).fetchone()[0]
                started = time.perf_counter()
                hits = flatten_results(recall.search(
                    request['question'], project=project, limit=10 if kind == 'locomo' else 5,
                    detail='full', record_usage=record_usage,
                ))
                elapsed = (time.perf_counter() - started) * 1000
                connection.send({'ms': elapsed, 'ids': [hit['id'] for hit in hits],
                                 'scores': [hit.get('rrf_score', hit.get('score')) for hit in hits]})
        finally:
            store.db.close()
            connection.close()


def receive(connection):
    if not connection.poll(WORKER_TIMEOUT_SECONDS):
        raise TimeoutError('Search benchmark worker did not respond')
    return connection.recv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('locomo', 'longmemeval'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record-usage', action='store_true')
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()
    rows = json.loads((ROOT / f'docs/benchmarks/context-v14/{args.kind}-contexts.json').read_text())
    rows = [rows[index] for index in np.linspace(0, len(rows) - 1, args.count, dtype=int)]
    raw = {}
    if args.kind == 'longmemeval':
        raw = {row['question_id']: row['question'] for row in json.loads((ROOT / 'benchmarks/data/longmemeval_s.json').read_text())}
    context = multiprocessing.get_context('spawn')
    processes, connections, measurements = {}, {}, []
    try:
        for name, code in (('before', ROOT / 'backups/optimization-v14-20260911'), ('after', ROOT)):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, code, args.kind, args.record_usage))
            process.start()
            child.close()
            processes[name], connections[name] = process, parent
        for connection in connections.values():
            assert receive(connection) == {'ready': True}
        for number, row in enumerate(rows):
            order = ('before', 'after') if number % 2 == 0 else ('after', 'before')
            for temperature in ('first', 'cold', 'warm'):
                result = {'id': row['id'], 'temperature': temperature}
                for name in order:
                    connection = connections[name]
                    connection.send({'temperature': temperature, 'anchor': row['anchor_ids'][0],
                                     'question': raw.get(row['id'], row['question'])})
                    result[name] = receive(connection)
                measurements.append(result)
            if (number + 1) % 25 == 0:
                sys.stderr.write(json.dumps({'completed': number + 1}) + '\n')
        summary = {}
        for temperature in ('first', 'cold', 'warm'):
            selected = [row for row in measurements if row['temperature'] == temperature]
            summary[temperature] = {
                name: {'p50_ms': float(np.percentile([row[name]['ms'] for row in selected], 50)),
                       'p95_ms': float(np.percentile([row[name]['ms'] for row in selected], 95))}
                for name in ('before', 'after')
            }
            ratios = np.array([row['after']['ms'] / row['before']['ms'] for row in selected])
            summary[temperature]['median_after_before_ratio'] = float(np.median(ratios))
        report = {'kind': args.kind, 'queries': len(rows), 'record_usage': args.record_usage,
                  'summary': summary, 'rows': measurements,
                  'id_differences': [row['id'] for row in measurements if row['before']['ids'] != row['after']['ids']],
                  'score_differences': [row['id'] for row in measurements if row['before']['scores'] != row['after']['scores']]}
        args.output.write_text(json.dumps(report, indent=2))
        sys.stdout.write(json.dumps({key: value for key, value in report.items() if key != 'rows'}) + '\n')
    finally:
        for name, connection in connections.items():
            process = processes[name]
            if process.is_alive():
                connection.send(None)
            connection.close()
        for process in processes.values():
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)


if __name__ == '__main__':
    main()
