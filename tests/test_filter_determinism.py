import json
import os
import subprocess
import sys
from pathlib import Path

from content_filter import run_pipeline


def test_preserved_values_keep_first_occurrence_order_after_normalization():
    text = (
        'https://example.test/z, https://example.test/a https://example.test/z.\n'
        '/var/z /var/a /var/z ~/z ~/a ~/z\n'
        '`zeta` `alpha` `zeta`'
    )
    assert run_pipeline(text, {'head': 0}) == (
        'preserved: https://example.test/z, https://example.test/a, '
        '/var/z, /var/a, ~/z, ~/a, `zeta`, `alpha`'
    )


def test_filtered_source_is_identical_across_python_hash_seeds():
    source = Path(__file__).resolve().parents[1] / 'src'
    program = (
        'import json; from content_filter import run_pipeline; '
        'print(json.dumps(run_pipeline("`red` `green` `blue` `cyan` `amber` `violet`", {"head": 0})))'
    )
    outputs = [subprocess.check_output(
        [sys.executable, '-c', program], text=True,
        env={**os.environ, 'PYTHONPATH': str(source), 'PYTHONHASHSEED': str(seed)},
    ) for seed in (1, 2, 3)]
    assert len(set(outputs)) == 1
    assert json.loads(outputs[0]) == 'preserved: `red`, `green`, `blue`, `cyan`, `amber`, `violet`'
