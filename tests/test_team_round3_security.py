"""14.10.0 round 3: lockout without a denial of service, opaque dashboard sessions, split pending-request caps."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from team_memory.accounts import (
    DEVICE_TTL_SECONDS,
    MAX_DEVICES_PER_USER,
    AccountPolicy,
    Accounts,
)
from team_memory.app import (
    MAX_PENDING_PUBLIC,
    MAX_PENDING_REQUESTS,
    MAX_PENDING_UNAUTHENTICATED,
    MAX_PUBLIC_PER_CLIENT,
    PENDING_PUBLIC_ENV,
    PUBLIC_PER_CLIENT_ENV,
    PENDING_ENV,
    PENDING_UNAUTHENTICATED_ENV,
    PendingLimits,
    RecentCredentials,
    create_app,
)
from team_memory.contracts import RateLimited, Unauthorized
from team_memory.dashboard import DEVICE_COOKIE
from team_memory.dashboard_service import DashboardService
from team_memory.metrics import Metrics
from team_memory.registry import Registry
from team_memory.service import MemoryService
from team_memory.settings import SettingsStore, load_cipher
from team_memory.worker import WorkerPool
from tests.team_db_helpers import identity, skip_unless_sqlite

PASSWORD = 'correct horse battery staple'
ROOT = Path(__file__).resolve().parents[1]
POLICY = AccountPolicy(max_failures=3, max_failures_per_account=6, lock_seconds=300)


@pytest.fixture(autouse=True)
def backend(team_backend):
    """Every test of this module runs on each selected team backend (--backend)."""
    return team_backend


class Clock:
    def __init__(self):
        self.now = 5_000_000.0

    def __call__(self):
        return self.now


def make_user(registry, accounts, user_id, role='member'):
    registry.add_user(user_id, user_id.title())
    if role != 'member':
        registry.set_org_role(user_id, role)
    return accounts.redeem_invite(user_id, accounts.issue_invite(user_id).code, PASSWORD, '127.0.0.1')


@pytest.fixture
def lone(tmp_path):
    clock = Clock()
    registry = Registry(tmp_path / 'root', clock=clock)
    accounts = Accounts(registry, POLICY, clock=clock)
    first = make_user(registry, accounts, 'root', 'superadmin')
    make_user(registry, accounts, 'dev')
    return SimpleNamespace(registry=registry, accounts=accounts, clock=clock, device=first.device_id)


def spray(accounts, user_id, count, prefix='198.51.100'):
    for index in range(count):
        with pytest.raises(Unauthorized):
            accounts.login_password(user_id, 'wrong password!', f'{prefix}.{index}')


# Item 1: lockout -------------------------------------------------------------------------------

def test_redeeming_an_invite_registers_the_browser_as_a_known_device(lone):
    assert lone.device
    again = lone.accounts.login_password('root', PASSWORD, '203.0.113.1', lone.device)
    assert again.device_id == lone.device
    fresh = lone.accounts.login_password('root', PASSWORD, '203.0.113.1')
    assert fresh.device_id and fresh.device_id != lone.device


def test_attacker_cannot_lock_the_owner_out_of_a_known_device(lone):
    spray(lone.accounts, 'root', 6)
    with pytest.raises(RateLimited):
        lone.accounts.login_password('root', PASSWORD, '192.0.2.50')
    with pytest.raises(RateLimited):
        lone.accounts.login_password('root', PASSWORD, '198.51.100.0')
    # The owner comes from a new address but with the browser that signed in before.
    assert lone.accounts.login_password('root', PASSWORD, '192.0.2.99', lone.device).actor.user_id == 'root'
    # ...and from an address the attacker already burned (the pair counter is locked too).
    assert lone.accounts.login_password('root', PASSWORD, '198.51.100.0', lone.device).actor.user_id == 'root'


def test_account_lock_still_stops_unknown_clients_and_expires(lone):
    spray(lone.accounts, 'root', 6)
    with pytest.raises(RateLimited):
        lone.accounts.login_password('root', PASSWORD, '192.0.2.50', 'forged-device-cookie')
    lone.clock.now += POLICY.lock_seconds + 1
    assert lone.accounts.login_password('root', PASSWORD, '192.0.2.50').actor.user_id == 'root'


def test_a_device_of_another_account_grants_nothing(lone):
    other = lone.accounts.login_password('dev', PASSWORD, '203.0.113.7').device_id
    spray(lone.accounts, 'root', 6)
    with pytest.raises(RateLimited):
        lone.accounts.login_password('root', PASSWORD, '192.0.2.50', other)


def test_known_device_is_throttled_by_its_own_counter(lone):
    for _ in range(POLICY.max_failures):
        with pytest.raises(Unauthorized):
            lone.accounts.login_password('root', 'wrong password!', '192.0.2.60', lone.device)
    with pytest.raises(RateLimited):
        lone.accounts.login_password('root', PASSWORD, '192.0.2.60', lone.device)
    # Its failures never fed the shared counters, so strangers are unaffected.
    assert lone.accounts.login_password('root', PASSWORD, '192.0.2.61').actor.user_id == 'root'


def test_expired_device_is_no_longer_trusted(lone):
    spray(lone.accounts, 'root', 6)
    lone.clock.now += DEVICE_TTL_SECONDS + 1
    spray(lone.accounts, 'root', 6, prefix='203.0.113')
    with pytest.raises(RateLimited):
        lone.accounts.login_password('root', PASSWORD, '192.0.2.99', lone.device)


def test_invite_redemption_recovers_a_locked_account(lone):
    spray(lone.accounts, 'root', 6)
    with pytest.raises(RateLimited):
        lone.accounts.login_password('root', PASSWORD, '192.0.2.50')
    invite = lone.accounts.issue_invite('root')
    session = lone.accounts.redeem_invite('root', invite.code, PASSWORD, '192.0.2.50')
    assert session.actor.user_id == 'root' and session.device_id


def test_unlock_clears_every_counter_of_the_account(lone):
    spray(lone.accounts, 'root', 6)
    for index in range(POLICY.max_failures):
        with pytest.raises(Unauthorized):
            lone.accounts.login_password('root', 'wrong password!', '192.0.2.60', lone.device)
    assert lone.accounts.unlock('root') >= 3
    assert lone.accounts.login_password('root', PASSWORD, '198.51.100.0').actor.user_id == 'root'
    assert lone.accounts.login_password('root', PASSWORD, '192.0.2.60', lone.device).actor.user_id == 'root'


def test_unlock_leaves_other_accounts_and_can_lift_an_address(lone):
    spray(lone.accounts, 'dev', 6)
    assert lone.accounts.unlock('root', ip='198.51.100.0') >= 1
    with pytest.raises(RateLimited):
        lone.accounts.login_password('dev', PASSWORD, '192.0.2.50')


def test_cli_login_unlock_recovers_a_locked_superadmin(lone):
    skip_unless_sqlite(lone.registry, 'the CLI is exercised against the SQLite root')
    spray(lone.accounts, 'root', 6)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    done = subprocess.run([sys.executable, '-m', 'team_memory.cli', '--root', str(lone.registry.root),
                           'login-unlock', 'root'], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert 'Cleared' in done.stdout
    assert lone.accounts.login_password('root', PASSWORD, '192.0.2.50').actor.user_id == 'root'


def test_disabling_a_user_forgets_their_devices(lone):
    lone.registry.set_active('dev', False)
    lone.registry.set_active('dev', True)
    with identity(lone.registry) as db:
        assert db.execute("SELECT COUNT(*) FROM known_devices WHERE user_id='dev'").fetchone()[0] == 0


def _device_rows(lone, user_id):
    with identity(lone.registry) as db:
        return db.execute("SELECT COUNT(*) FROM known_devices WHERE user_id=?", (user_id,)).fetchone()[0]


def test_changing_the_password_forgets_every_device(lone):
    assert _device_rows(lone, 'root') == 1
    session = lone.accounts.login_password('root', PASSWORD, '203.0.113.1', lone.device)
    lone.accounts.change_password(session.actor, PASSWORD, 'another long passphrase!', session.session_id,
                                  '203.0.113.1', lone.device)
    assert _device_rows(lone, 'root') == 0
    # The stolen cookie is now an unknown client: it falls under the shared address, pair and account limits.
    spray(lone.accounts, 'root', 6)
    with pytest.raises(RateLimited):
        lone.accounts.login_password('root', 'another long passphrase!', '192.0.2.77', lone.device)


def test_a_reset_invite_forgets_every_device(lone):
    assert _device_rows(lone, 'dev') == 1
    reset = lone.accounts.issue_invite('dev')
    session = lone.accounts.redeem_invite('dev', reset.code, 'a brand new passphrase!', '127.0.0.1')
    with identity(lone.registry) as db:
        digests = [row[0] for row in db.execute("SELECT digest FROM known_devices WHERE user_id='dev'").fetchall()]
    assert digests == [Registry.digest(session.device_id)]


def test_devices_per_user_are_capped(lone):
    for _ in range(MAX_DEVICES_PER_USER + 3):
        lone.clock.now += 1
        lone.accounts.login_password('root', PASSWORD, '203.0.113.1')
    with identity(lone.registry) as db:
        assert db.execute("SELECT COUNT(*) FROM known_devices WHERE user_id='root'").fetchone()[0] == MAX_DEVICES_PER_USER


def build_http(root):
    registry = Registry(root)
    accounts = Accounts(registry, POLICY)
    settings = SettingsStore(registry, load_cipher(root, {}), {})
    pool = WorkerPool(registry.root, maximum=2, environment=settings.overrides)
    dashboard = DashboardService(registry, accounts, MemoryService(registry, pool), settings, Metrics())
    return registry, accounts, create_app(dashboard.memory, dashboard)


def test_http_owner_keeps_signing_in_while_an_attacker_locks_the_account(tmp_path):
    registry, accounts, app = build_http(tmp_path / 'root')
    make_user(registry, accounts, 'root', 'superadmin')
    with TestClient(app) as client:
        first = client.post('/dashboard/api/login', json={'user_id': 'root', 'password': PASSWORD})
        assert first.status_code == 200 and DEVICE_COOKIE in client.cookies
        assert 'httponly' in first.headers['set-cookie'].lower()
        owner_device = client.cookies.get(DEVICE_COOKIE)
        client.cookies.clear()
        for _ in range(POLICY.max_failures_per_account):
            assert client.post('/dashboard/api/login', json={'user_id': 'root', 'password': 'wrong password!'}
                               ).status_code in (401, 429)
        assert client.post('/dashboard/api/login', json={'user_id': 'root', 'password': PASSWORD}).status_code == 429
        client.cookies.set(DEVICE_COOKIE, owner_device)
        again = client.post('/dashboard/api/login', json={'user_id': 'root', 'password': PASSWORD})
        assert again.status_code == 200, again.text


# Item 2: opaque server-side sessions ---------------------------------------------------------------

def test_dashboard_cookie_is_an_opaque_session_hashed_at_rest(tmp_path):
    registry, accounts, app = build_http(tmp_path / 'root')
    make_user(registry, accounts, 'dev')
    token = registry.issue_token('dev', 'agent')
    with TestClient(app) as client:
        reply = client.post('/dashboard/api/login/token', json={'token': token})
        assert reply.status_code == 200
        cookie = client.cookies.get('tam_session')
        assert cookie and cookie != token and token not in reply.headers['set-cookie']
        assert DEVICE_COOKIE not in client.cookies
        with identity(registry) as db:
            stored = [row[0] for row in db.execute('SELECT digest FROM sessions').fetchall()]
        assert Registry.digest(cookie) in stored and cookie not in stored
        assert client.get('/dashboard/api/session').status_code == 200
        registry.revoke(token)
        assert client.get('/dashboard/api/session').status_code == 401


def test_dashboard_logout_invalidates_a_token_opened_session(tmp_path):
    registry, accounts, app = build_http(tmp_path / 'root')
    make_user(registry, accounts, 'dev')
    token = registry.issue_token('dev', 'agent')
    with TestClient(app) as client:
        csrf = client.post('/dashboard/api/login/token', json={'token': token}).json()['csrf']
        stale = client.cookies.get('tam_session')
        assert client.post('/dashboard/api/logout', headers={'X-CSRF-Token': csrf}).status_code == 200
        client.cookies.set('tam_session', stale)
        assert client.get('/dashboard/api/session').status_code == 401


# Item 4: pending caps ------------------------------------------------------------------------------

@pytest.fixture
def world(tmp_path):
    registry, accounts, app = build_http(tmp_path / 'root')
    make_user(registry, accounts, 'root', 'superadmin')
    with TestClient(app) as client:
        yield SimpleNamespace(registry=registry, accounts=accounts, client=client, app=app)


def test_anonymous_flood_does_not_deny_a_known_agent(world):
    token = world.registry.issue_token('root', 'agent')
    headers = {'Authorization': 'Bearer ' + token}
    authentication = world.app.app
    assert world.client.post('/api/call', headers=headers, json={'name': 'memory_scopes'}).status_code == 200
    authentication.unauthenticated = MAX_PENDING_UNAUTHENTICATED
    assert world.client.post('/api/call', headers={'Authorization': 'Bearer ' + 'B' * 43}, json={}).status_code == 429
    # Static files and sign-in need no lookup: they have their own pool and are not blocked by the anonymous one.
    assert world.client.get('/dashboard/static/missing.js').status_code != 429
    assert world.client.post('/api/call', headers=headers, json={'name': 'memory_scopes'}).status_code == 200


def test_signed_in_dashboard_user_survives_an_anonymous_flood(world):
    reply = world.client.post('/dashboard/api/login', json={'user_id': 'root', 'password': PASSWORD})
    assert reply.status_code == 200
    assert world.client.get('/dashboard/api/session').status_code == 200
    world.app.app.unauthenticated = MAX_PENDING_UNAUTHENTICATED
    world.app.app.public = MAX_PENDING_PUBLIC
    assert world.client.get('/dashboard/api/session').status_code == 200
    world.client.cookies.clear()
    assert world.client.post('/dashboard/api/login', json={'user_id': 'x', 'password': 'y'}).status_code == 429


def test_authenticated_cap_is_counted_after_authentication(world):
    token = world.registry.issue_token('root', 'agent')
    authentication = world.app.app
    authentication.pending = authentication.limits.authenticated
    ok = world.client.post('/api/call', headers={'Authorization': 'Bearer ' + token}, json={'name': 'memory_scopes'})
    assert ok.status_code == 429
    # A wrong token is still answered 401 (anonymous pool), it does not need an authenticated slot.
    bad = world.client.post('/api/call', headers={'Authorization': 'Bearer ' + 'C' * 43}, json={})
    assert bad.status_code == 401
    authentication.pending = 0
    assert authentication.unauthenticated == 0


def test_anonymous_slot_is_released_after_every_request(world):
    for _ in range(MAX_PENDING_UNAUTHENTICATED + 3):
        assert world.client.post('/api/call', headers={'Authorization': 'Bearer ' + 'D' * 43}, json={}).status_code == 401
    assert world.app.app.unauthenticated == 0


def test_pending_limits_come_from_the_environment():
    limits = PendingLimits.from_env({PENDING_ENV: '50', PENDING_UNAUTHENTICATED_ENV: '3'})
    assert (limits.authenticated, limits.unauthenticated) == (50, 3)
    assert PendingLimits.from_env({}) == PendingLimits(MAX_PENDING_REQUESTS, MAX_PENDING_UNAUTHENTICATED,
                                                       MAX_PENDING_PUBLIC, MAX_PUBLIC_PER_CLIENT)
    tuned = PendingLimits.from_env({PENDING_PUBLIC_ENV: '5', PUBLIC_PER_CLIENT_ENV: '2'})
    assert (tuned.public, tuned.public_per_client) == (5, 2)
    with pytest.raises(ValueError, match='positive'):
        PendingLimits.from_env({PUBLIC_PER_CLIENT_ENV: '0'})
    assert MAX_PENDING_UNAUTHENTICATED < MAX_PENDING_REQUESTS
    with pytest.raises(ValueError, match='positive'):
        PendingLimits.from_env({PENDING_UNAUTHENTICATED_ENV: '0'})


def test_recent_credentials_expire_and_stay_bounded():
    now = [100.0]
    recent = RecentCredentials(clock=lambda: now[0], ttl=10, capacity=2)
    recent.remember('bearer', 'a')
    assert recent.known('bearer', 'a') and not recent.known('session', 'a')
    now[0] += 11
    assert not recent.known('bearer', 'a')
    recent.remember('bearer', 'b')
    recent.remember('bearer', 'c')
    recent.remember('bearer', 'd')
    assert len(recent._seen) <= 2


def asgi_request(app, path, *, method='POST', client='203.0.113.1', headers=None, body=b'', stall=None):
    """One request straight through the ASGI stack; `stall` is an Event the body waits for."""
    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method, 'path': path,
             'raw_path': path.encode(), 'query_string': b'', 'root_path': '', 'scheme': 'http',
             'headers': [(b'host', b'testserver'), *[(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]],
             'client': (client, 5000), 'server': ('testserver', 80)}
    sent = []

    async def receive():
        if stall is not None:
            await stall.wait()
        return {'type': 'http.request', 'body': body, 'more_body': False}

    async def send(message):
        sent.append(message)

    async def run():
        await app(scope, receive, send)
        return next(m['status'] for m in sent if m['type'] == 'http.response.start')
    return run()


def test_slow_anonymous_bodies_block_neither_agents_nor_other_sign_ins(world):
    token = world.registry.issue_token('root', 'agent')
    authentication = world.app.app
    call = json.dumps({'name': 'memory_scopes'}).encode()
    login = json.dumps({'user_id': 'root', 'password': PASSWORD}).encode()

    async def scenario():
        stall = asyncio.Event()
        # One client opens more stalled sign-ins than its share: the excess is refused at once.
        same = [asyncio.create_task(asgi_request(authentication, '/dashboard/api/login', stall=stall, body=login))
                for _ in range(MAX_PUBLIC_PER_CLIENT + 2)]
        # Eight other clients stall too, as many as the anonymous pool has slots.
        others = [asyncio.create_task(asgi_request(authentication, '/dashboard/api/invite/redeem', client=f'198.51.100.{n}',
                                                   stall=stall, body=login)) for n in range(MAX_PENDING_UNAUTHENTICATED)]
        await asyncio.sleep(0.2)
        assert authentication.public == MAX_PUBLIC_PER_CLIENT + MAX_PENDING_UNAUTHENTICATED
        assert authentication.unauthenticated == 0
        # An agent whose token was last seen long ago still gets in, and so does another person's sign-in.
        authentication.recent._seen.clear()
        agent = await asgi_request(authentication, '/api/call', headers={'Authorization': 'Bearer ' + token}, body=call)
        other_user = await asgi_request(authentication, '/dashboard/api/login', client='192.0.2.9', body=login,
                                         headers={'Content-Type': 'application/json'})
        stall.set()
        await asyncio.gather(*same, *others)
        return agent, other_user, [task.result() for task in same]

    agent, other_user, same = asyncio.run(scenario())
    assert agent == 200
    assert other_user == 200
    assert sorted(same).count(429) == 2
    assert authentication.public == 0 and not authentication.public_clients


def test_anonymous_body_times_out_quickly(world, monkeypatch):
    monkeypatch.setattr('team_memory.app.PUBLIC_READ_TIMEOUT_SECONDS', 0.05)
    status = asyncio.run(asgi_request(world.app.app, '/dashboard/api/login', stall=asyncio.Event()))
    assert status == 408
    assert world.app.app.public == 0 and not world.app.app.public_clients


def test_recent_credentials_default_covers_an_agents_idle_gap():
    from team_memory.app import RECENT_CREDENTIAL_SECONDS
    assert RECENT_CREDENTIAL_SECONDS >= 120
