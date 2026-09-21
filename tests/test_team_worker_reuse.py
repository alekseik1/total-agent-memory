import pytest

from team_memory.registry import Registry
from team_memory.service import MemoryService
from team_memory.worker import WorkerPool


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('maximum', [None, 2])
async def test_three_scopes_reuse_warm_workers(tmp_path, monkeypatch, maximum):
    monkeypatch.setenv('MEMORY_LLM_ENABLED', 'false')
    monkeypatch.setenv('MEMORY_QUALITY_GATE_ENABLED', 'false')
    registry = Registry(tmp_path)
    registry.add_user('vasya', 'Вася')
    registry.add_team('engineering', 'Разработка')
    registry.membership('vasya', 'engineering', 'editor')
    token = registry.issue_token('vasya', 'reuse-test')
    pool = WorkerPool(tmp_path) if maximum is None else WorkerPool(tmp_path, maximum=maximum)
    service = MemoryService(registry, pool)
    try:
        scopes = registry.workspaces(registry.authenticate(token))
        for workspace in scopes:
            await service.call(token, 'memory_save', {
                'scope': workspace.scope.model_dump(mode='json'),
                'content': f'Launch procedure for {workspace.scope.kind.value}: check the safety log.',
            })
        processes = {key: worker[0].pid for key, worker in pool.workers.items()}
        assert len(processes) == (len(scopes) if maximum is None else maximum)
        for query in ('launch procedure', 'safety log'):
            result = await service.call(token, 'memory_recall', {'query': query})
            assert {item['scope']['kind'] for item in result['results']} == {'personal', 'team', 'shared'}
            current = {key: worker[0].pid for key, worker in pool.workers.items()}
            if maximum is None:
                assert current == processes
            else:
                assert len(set(current.values()) & set(processes.values())) == maximum - 1
            processes = current
    finally:
        pool.close()


@pytest.mark.anyio
async def test_worker_execution_order_does_not_change_search_ties(tmp_path, monkeypatch):
    registry = Registry(tmp_path)
    registry.add_user('vasya', 'Вася')
    registry.add_team('engineering', 'Разработка')
    registry.membership('vasya', 'engineering', 'reader')
    token = registry.issue_token('vasya', 'ordering-test')
    pool = WorkerPool(tmp_path)
    expected = [workspace.scope.model_dump(mode='json')
                for workspace in registry.workspaces(registry.authenticate(token))]
    monkeypatch.setattr(pool, 'search_order', lambda scopes: list(reversed(scopes)))
    monkeypatch.setattr(pool, 'invoke', lambda work, credential: [{'id': 1, 'score': 0.5}])
    result = await MemoryService(registry, pool).call(token, 'memory_recall', {'query': 'launch safety'})
    assert [item['scope'] for item in result['results']] == expected
