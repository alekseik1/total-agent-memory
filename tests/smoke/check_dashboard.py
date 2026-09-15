import json
import os
import time
import urllib.error
import urllib.request
from importlib.metadata import version

START_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 3
DEFAULT_PORT = 37737


def main():
    expected = version('total-agent-memory')
    port = int(os.environ.get('DASHBOARD_PORT', DEFAULT_PORT))
    url = f'http://127.0.0.1:{port}/api/release'
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while True:
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                release = json.load(response)
            assert release['name'] == 'total-agent-memory'
            assert release['version'] == expected, release
            assert release['release_date']
            return
        except (OSError, urllib.error.URLError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == '__main__':
    main()
