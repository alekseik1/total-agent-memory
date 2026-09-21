import argparse
import asyncio
import hashlib
import json
import os
import resource
import statistics
import tempfile
import time
from pathlib import Path

from team_memory.registry import Registry
from team_memory.service import MemoryService
from team_memory.worker import WorkerPool

QUERIES = ('launch procedure', 'safety log', 'launch safety', 'procedure log')
IDLE_SECONDS = 5


def cpu_seconds(pool):
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    total = usage.ru_utime + usage.ru_stime
    ticks = os.sysconf('SC_CLK_TCK')
    for process, _ in pool.workers.values():
        values = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
        total += (int(values[11]) + int(values[12])) / ticks
    return total


def worker_rss_sum_mib(pool):
    kib = 0
    for process, _ in pool.workers.values():
        fields = Path(f'/proc/{process.pid}/status').read_text().splitlines()
        kib += int(next(line.split()[1] for line in fields if line.startswith('VmRSS:')))
    return kib / 1024


async def profile(maximum):
    with tempfile.TemporaryDirectory(prefix='tam-cpu-profile-') as directory:
        root = Path(directory)
        registry = Registry(root)
        registry.add_user('vasya', 'Вася')
        registry.add_team('engineering', 'Разработка')
        registry.membership('vasya', 'engineering', 'editor')
        token = registry.issue_token('vasya', 'cpu-profile')
        pool = WorkerPool(root, maximum=maximum)
        service = MemoryService(registry, pool)
        try:
            for workspace in registry.workspaces(registry.authenticate(token)):
                await service.call(token, 'memory_save', {
                    'scope': workspace.scope.model_dump(mode='json'),
                    'content': f'Launch procedure for {workspace.scope.kind.value}: check the safety log.',
                })
            before = cpu_seconds(pool)
            timings, results, retained = [], [], []
            for query in QUERIES:
                previous = {worker[0].pid for worker in pool.workers.values()}
                started = time.perf_counter()
                result = await service.call(token, 'memory_recall', {'query': query})
                timings.append((time.perf_counter() - started) * 1000)
                retained.append(len(previous & {worker[0].pid for worker in pool.workers.values()}))
                results.append([(item['scope'], item['record']['id'], item['record']['content'])
                                for item in result['results']])
            query_cpu = cpu_seconds(pool) - before
            before = cpu_seconds(pool)
            await asyncio.sleep(IDLE_SECONDS)
            return {'maximum': maximum, 'queries': len(QUERIES), 'query_cpu_seconds': query_cpu,
                    'query_p50_ms': statistics.median(timings), 'query_max_ms': max(timings),
                    'query_samples_ms': timings, 'retained_workers_per_query': retained,
                    'worker_rss_sum_mib': worker_rss_sum_mib(pool),
                    'idle_seconds': IDLE_SECONDS, 'idle_worker_cpu_seconds': cpu_seconds(pool) - before,
                    'results_sha256': hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()}
        finally:
            pool.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(asyncio.run(profile(args.workers)), indent=2) + '\n')
