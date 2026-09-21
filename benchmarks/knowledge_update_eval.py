#!/usr/bin/env python3
"""Knowledge-update evaluation: does memory answer with the *current* value of a fact?

Two suites run through the shipping in-process path (`Store.save_knowledge`,
`Recall.search`, `answer_endpoint.answer_response`) on a throwaway database:

  synthetic    — benchmarks/data/knowledge_update_scenarios.json (ru + en), mixed
                 into one project per language together with deterministic noise
                 records about other people. Record ages are stamped into
                 created_at, as if the facts were saved months apart.
  longmemeval  — the knowledge-update category of LongMemEval-S. Each question's
                 haystack goes into its own project; every session is stamped
                 with its own haystack date.

Answers are graded by the same gpt-4.1-mini judge used by the v14 QA studies
(~/PROJECT/judge-probe/research/locomo_qa_budget.py) under a spending ledger.

Usage:
    python benchmarks/knowledge_update_eval.py --suite synthetic --out results/ku-base
    python benchmarks/knowledge_update_eval.py --suite longmemeval --out results/ku-base
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / 'benchmarks/data/knowledge_update_scenarios.json'
LONGMEMEVAL = ROOT / 'benchmarks/data/longmemeval_s.json'
JUDGE_DIR = Path.home() / 'PROJECT/judge-probe'
SERVICE_PLIST = Path.home() / 'Library/LaunchAgents/com.total-agent-memory.reflection.plist'
NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
NOISE_PER_LANGUAGE = 300
NOISE_SEED = 20260921
TOP_K = 5
REFUSAL_PREFIX = 'Not enough information'

NOISE_NAMES = {
    'ru': ['Антон', 'Борис', 'Галина', 'Денис', 'Евгения', 'Жанна', 'Зоя', 'Кирилл', 'Лариса', 'Максим',
           'Никита', 'Полина', 'Роман', 'Светлана', 'Тарас', 'Ульяна', 'Фёдор', 'Юлия', 'Яна', 'Глеб'],
    'en': ['Adam', 'Beth', 'Carl', 'Diana', 'Ethan', 'Fiona', 'Grace', 'Henry', 'Irene', 'Jack',
           'Kevin', 'Laura', 'Mike', 'Nora', 'Owen', 'Quinn', 'Ruth', 'Steve', 'Tina', 'Victor'],
}
NOISE_TEMPLATES = {
    'ru': [('{n} любит {v} цвет.', ['оранжевый', 'серый', 'бордовый', 'голубой', 'чёрный', 'белый']),
           ('{n} живёт в городе {v}.', ['Самара', 'Пермь', 'Тверь', 'Сочи', 'Омск', 'Уфа']),
           ('{n} работает в компании {v}.', ['Авито', 'Тинькофф', 'VK', 'Лаборатория Касперского', 'МТС']),
           ('{n} ездит на {v}.', ['Лада Веста', 'Skoda Octavia', 'Hyundai Solaris', 'велосипеде', 'метро']),
           ('{n} по утрам пьёт {v}.', ['какао', 'чёрный чай', 'капучино', 'смузи', 'воду с лимоном']),
           ('{n} увлекается {v}.', ['шахматами', 'бегом', 'фотографией', 'йогой', 'рыбалкой']),
           ('{n} недавно перестал(а) {v}.', ['курить', 'есть сладкое', 'смотреть сериалы', 'играть в приставку'])],
    'en': [('{n} loves the color {v}.', ['orange', 'gray', 'maroon', 'sky blue', 'black', 'white']),
           ('{n} lives in {v}.', ['Austin', 'Chicago', 'Miami', 'Phoenix', 'Atlanta', 'Dallas']),
           ('{n} works at {v}.', ['Netflix', 'Shopify', 'Airbnb', 'Dropbox', 'Uber']),
           ('{n} drives a {v}.', ['Honda Civic', 'Ford Focus', 'Tesla Model 3', 'bicycle', 'Subaru Outback']),
           ('{n} drinks {v} in the morning.', ['cocoa', 'black tea', 'a cappuccino', 'a smoothie', 'lemon water']),
           ('{n} is into {v}.', ['chess', 'running', 'photography', 'yoga', 'fishing']),
           ('{n} recently stopped {v}.', ['smoking', 'eating sweets', 'binge-watching shows', 'gaming'])],
}


def configure_environment(db_dir: str) -> None:
    import plistlib

    service_env = plistlib.loads(SERVICE_PLIST.read_bytes()).get('EnvironmentVariables', {})
    for name in ('MEMORY_LLM_PROVIDER', 'MEMORY_LLM_MODEL', 'MEMORY_LLM_API_KEY'):
        if not os.environ.get(name):
            if not service_env.get(name):
                raise SystemExit(f'{name} is not set and not found in {SERVICE_PLIST}')
            os.environ[name] = service_env[name]
    os.environ['TAM_MEMORY_DIR'] = db_dir
    os.environ['CLAUDE_MEMORY_DIR'] = db_dir
    os.environ['MEMORY_LLM_ENABLED'] = 'false'
    os.environ.setdefault('MEMORY_QUIET', '1')
    sys.path[:0] = [str(ROOT / 'src'), str(JUDGE_DIR / 'research')]


class Harness:
    def __init__(self) -> None:
        import server

        self.store = server.Store()
        self.recall = server.Recall(self.store)

    def save(self, project: str, text: str, created_at: datetime, tags: list[str]) -> tuple[int, bool]:
        sid = f'ku__{project}'
        self.store.session_start(sid, project=project)
        rid, deduplicated, *_ = self.store.save_knowledge(
            sid=sid, content=text, ktype='fact', project=project, tags=tags, skip_quality=True)
        stamp = created_at.isoformat().replace('+00:00', 'Z')
        self.store.db.execute('UPDATE knowledge SET created_at=?, last_confirmed=? WHERE id=?', (stamp, stamp, rid))
        self.store.db.commit()
        return rid, deduplicated

    def ranked_ids(self, query: str, project: str) -> list[int]:
        from memory_core.retrieval import flatten_results

        result = self.recall.search(query=query, project=project, limit=TOP_K, detail='summary', record_usage=False)
        return [hit['id'] for hit in flatten_results(result)][:TOP_K]

    def answer(self, query: str, project: str) -> dict:
        from answer_endpoint import answer_response

        started = time.perf_counter()
        try:
            result = answer_response(self.store, self.recall, {'query': query, 'project': project})
        except Exception as error:  # noqa: BLE001 — every failure is recorded per question, never hidden
            return {'answer': None, 'error': f'{type(error).__name__}: {error}',
                    'elapsed_ms': (time.perf_counter() - started) * 1000}
        negative = result.negative
        verification = result.verification
        return {'answer': result.answer, 'error': None, 'status': result.draft.status,
                'draft_answer': result.draft.answer, 'rejection': result.draft.rejection,
                'verified': verification.supported if verification else None,
                'verification_reason': verification.reason if verification else None,
                'negative_decision': negative.decision if negative else None,
                'evidence_ids': [hit['id'] for hit in result.evidence],
                'elapsed_ms': (time.perf_counter() - started) * 1000}


class Judge:
    def __init__(self, ledger: Path, budget_usd: float):
        from locomo_qa_budget import JUDGE, BudgetClient, load_key

        self.prompt = JUDGE
        self.client = BudgetClient(load_key(JUDGE_DIR / '.env'), ledger, budget_usd)

    def grade(self, question: str, reference: str, candidate: str | None) -> tuple[bool, float]:
        if not candidate:
            return False, 0.0
        verdict = self.client.complete(self.prompt, json.dumps(
            {'question': question, 'reference': reference, 'candidate': candidate}, ensure_ascii=False))
        return verdict.text.strip().upper() == 'YES', verdict.cost_usd


def noise_records(lang: str) -> list[tuple[str, int]]:
    rng = random.Random(f'{NOISE_SEED}:{lang}')
    names, templates = NOISE_NAMES[lang], NOISE_TEMPLATES[lang]
    seen: set[str] = set()
    records: list[tuple[str, int]] = []
    while len(records) < NOISE_PER_LANGUAGE:
        template, values = rng.choice(templates)
        text = template.format(n=rng.choice(names), v=rng.choice(values))
        if text not in seen:
            seen.add(text)
            records.append((text, rng.randint(1, 365)))
    return records


class Sink:
    """Appends each graded row as soon as it exists, so a crash or a stop loses nothing."""

    def __init__(self, path: Path):
        self.path = path
        self.done = {json.loads(line)['id'] for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()

    def write(self, row: dict) -> None:
        with self.path.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def run_synthetic(harness: Harness, judge: Judge, only: set[str] | None, sink: Sink) -> None:
    scenarios = json.loads(SCENARIOS.read_text())['scenarios']
    if only:
        scenarios = [row for row in scenarios if row['id'] in only]
    if sink.done:
        raise SystemExit(f'{sink.path} already has answers; the synthetic suite shares one database, rerun into a new --out')
    for lang in sorted({row['lang'] for row in scenarios}):
        project = f'ku_synthetic_{lang}'
        for text, age in noise_records(lang):
            harness.save(project, text, NOW - timedelta(days=age), ['ku:noise'])
    rows = []
    for scenario in scenarios:
        project = f'ku_synthetic_{scenario["lang"]}'
        ids, dedup = [], []
        for record in sorted(scenario['records'], key=lambda item: -item['age_days']):
            rid, was_dedup = harness.save(project, record['text'], NOW - timedelta(days=record['age_days']),
                                          [f'ku:{scenario["id"]}'])
            ids.append(rid)
            dedup.append(was_dedup)
        rows.append({'scenario': scenario, 'project': project, 'record_ids': ids, 'deduplicated': dedup})
    for row in rows:
        scenario, (old_id, new_id) = row['scenario'], row['record_ids']
        ranked = harness.ranked_ids(scenario['question'], row['project'])
        answer = harness.answer(scenario['question'], row['project'])
        # Every synthetic reference states a fact, so a refusal is wrong without asking the judge.
        refused = (answer['answer'] or '').startswith(REFUSAL_PREFIX)
        correct, cost = (False, 0.0) if refused else judge.grade(scenario['question'], scenario['reference'], answer['answer'])
        sink.write({
            'id': scenario['id'], 'lang': scenario['lang'], 'kind': scenario['kind'],
            'question': scenario['question'], 'reference': scenario['reference'],
            'deduplicated': row['deduplicated'], 'recall_ids': ranked,
            'old_in_top': old_id in ranked, 'new_in_top': new_id in ranked,
            'new_above_old': new_id in ranked and (old_id not in ranked or ranked.index(new_id) < ranked.index(old_id)),
            'new_in_evidence': new_id in answer.get('evidence_ids', []),
            **answer, 'correct': correct, 'judge_cost_usd': cost})
        print(f'[synthetic] {scenario["id"]:6} {scenario["kind"]:13} correct={correct} '
              f'answer={(answer["answer"] or answer["error"] or "")[:90]!r}', flush=True)


def _session_date(raw: str) -> datetime:
    return datetime.strptime(raw.split(' (')[0] + ' ' + raw.split(') ')[1], '%Y/%m/%d %H:%M').replace(tzinfo=UTC)


def select_longmemeval(types: list[str], per_type: int) -> list[dict]:
    rows = json.loads(LONGMEMEVAL.read_text())
    selected: list[dict] = []
    for question_type in types:
        matching = [row for row in rows if row['question_type'] == question_type]
        if not matching:
            raise SystemExit(f'No LongMemEval questions of type {question_type!r}')
        selected.extend(matching[:per_type] if per_type else matching)
    return selected


def run_longmemeval(harness: Harness, judge: Judge, limit: int, only: set[str] | None, sink: Sink,
                    shard: tuple[int, int], types: list[str], per_type: int) -> None:
    data = select_longmemeval(types, per_type)
    if only:
        data = [row for row in data if row['question_id'] in only]
    data = data[:limit] if limit else data
    index, count = shard
    for entry in data[index::count]:
        if entry['question_id'] in sink.done:
            continue
        project = f'lme_{entry["question_id"]}'
        session_ids: dict[int, str] = {}
        for sid, session, date in zip(entry['haystack_session_ids'], entry['haystack_sessions'],
                                      entry['haystack_dates'], strict=True):
            text = '\n'.join(f'{turn["role"]}: {turn["content"]}' for turn in session if turn['content'].strip())
            if text:
                rid, _ = harness.save(project, f'[{date}]\n{text}', _session_date(date), [f'lme:{sid}'])
                session_ids[rid] = sid
        question = f'As of {entry["question_date"]}: {entry["question"]}'
        gold = set(entry['answer_session_ids'])
        ranked = [session_ids.get(rid) for rid in harness.ranked_ids(entry['question'], project)]
        answer = harness.answer(question, project)
        correct, cost = judge.grade(question, str(entry['answer']), answer['answer'])
        sink.write({
            'id': entry['question_id'], 'lang': 'en',
            'kind': 'abstention' if entry['question_id'].endswith('_abs') else entry['question_type'],
            'question': question, 'reference': str(entry['answer']),
            'recall_any': bool(gold & set(ranked)), 'recall_all': gold <= set(ranked),
            'evidence_sessions': sorted({session_ids[rid] for rid in answer.get('evidence_ids', []) if rid in session_ids}),
            **answer, 'correct': correct, 'judge_cost_usd': cost})
        print(f'[longmemeval] {entry["question_id"]:14} correct={correct} '
              f'answer={(answer["answer"] or answer["error"] or "")[:90]!r}', flush=True)


def summarize(results: list[dict]) -> dict:
    def block(rows: list[dict]) -> dict:
        out = {'n': len(rows), 'correct': sum(row['correct'] for row in rows),
               'errors': sum(row['error'] is not None for row in rows),
               'refusals': sum((row['answer'] or '').startswith(REFUSAL_PREFIX) for row in rows)}
        for key in ('new_in_top', 'new_above_old', 'new_in_evidence', 'recall_any', 'recall_all'):
            if rows and key in rows[0]:
                out[key] = sum(row[key] for row in rows)
        return out

    groups: dict[str, list[dict]] = {}
    for row in results:
        groups.setdefault(f'{row["lang"]}:{row["kind"]}', []).append(row)
        groups.setdefault(f'kind:{row["kind"]}', []).append(row)
        groups.setdefault(f'lang:{row["lang"]}', []).append(row)
    return {'overall': block(results), 'groups': {key: block(rows) for key, rows in sorted(groups.items())},
            'judge_cost_usd': round(sum(row['judge_cost_usd'] for row in results), 6),
            'median_answer_ms': sorted(row['elapsed_ms'] for row in results)[len(results) // 2] if results else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', choices=('synthetic', 'longmemeval'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--only', default='', help='comma-separated scenario or question ids')
    parser.add_argument('--judge-budget-usd', type=float, default=1.0)
    parser.add_argument('--shard', default='0/1', help='i/n: run every n-th LongMemEval question starting at i')
    parser.add_argument('--types', default='knowledge-update', help='comma-separated LongMemEval question types')
    parser.add_argument('--per-type', type=int, default=0, help='first N questions of each type (0 = all)')
    parser.add_argument('--summarize', action='store_true', help='only merge existing answer files into the summary')
    args = parser.parse_args()
    index, count = (int(part) for part in args.shard.split('/'))
    if not 0 <= index < count:
        raise SystemExit('--shard must be i/n with 0 <= i < n')

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.summarize:
        configure_environment(tempfile.mkdtemp(prefix=f'ku-{args.suite}-'))
        harness, judge = Harness(), Judge(args.out / f'{args.suite}-judge-ledger.db', args.judge_budget_usd)
        only = {item for item in args.only.split(',') if item} or None
        if args.suite == 'synthetic':
            run_synthetic(harness, judge, only, Sink(args.out / 'synthetic-answers.jsonl'))
        else:
            run_longmemeval(harness, judge, args.limit, only,
                            Sink(args.out / f'longmemeval-answers.shard{index}of{count}.jsonl'), (index, count),
                            [item for item in args.types.split(',') if item], args.per_type)
    results = [json.loads(line) for path in sorted(args.out.glob(f'{args.suite}-answers*.jsonl'))
               for line in path.read_text().splitlines() if line.strip()]
    summary = {'suite': args.suite, 'label': args.label, 'model': os.environ.get('MEMORY_LLM_MODEL'),
               'provider': os.environ.get('MEMORY_LLM_PROVIDER'), 'run_at': datetime.now(UTC).isoformat(),
               **summarize(results)}
    (args.out / f'{args.suite}-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary['overall'], ensure_ascii=False))


if __name__ == '__main__':
    main()
