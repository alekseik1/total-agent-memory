# Кандидат 14.0.0 — финальная проверка 2026-09-15

**DONE_WITH_CONCERNS.** Wheel, исходный архив и Docker-образы подготовлены.
Git, публикация пакетов и production deployment не выполнялись.
Быстрый режим остаётся режимом по умолчанию. Топ-10 не подтверждён,
а необязательный BGE на проверенном CPU не укладывается в200мс.

## Что доведено в этом продолжении

- Противоречивое grounded-утверждение `supported/inferred` вместе с `missing`
  после ограниченной попытки исправления возвращает явный отказ с причиной.
  Повреждённый протокол по-прежнему вызывает ошибку. Две регрессии сначала
  воспроизвели сбой, затем прошли; проверки точных цитат сохранены.
- Бюджет PyTorch задаётся до импорта библиотеки и применяется к OpenMP/BLAS.
  Настройка охватывает rerankers, NLI, ST embeddings, reembed и calibration.
  Лёгкий импорт и установка без необязательного Torch сохранены.
- Исправлена комплектность sdist: включены общие pytest fixtures, тестовые
  данные, requirements, установщики, launchagents, bootstrap, примеры,
  вспомогательные команды и файлы для Docker-сборки.
- В CI добавлен обязательный `source-archive` gate: полный pytest запускается
  из распакованного архива. Новые CPU/grounded регрессии включены в native matrix.

## Проверки

| Проверка | Результат |
|---|---|
| Полный Docker suite из рабочего дерева | **2162 passed,32 skipped**,189,19с |
| Полный suite из исходного архива | **2145 passed,49 skipped**,189,90с |
| Окружение с тяжёлыми моделями | **74 passed,4 skipped,2 deselected**,13,00с |
| macOS arm64,Python3.12.14 | wheel install/pipcheck/runtime; **26 passed,2 skipped** |
| Ubuntu24.04 aarch64 без Docker,Python3.12.3 | wheel install/pipcheck/runtime; **26 passed,2 skipped** |
| Windows11 ARM64,CPython3.12.14 x64 под эмуляцией | wheel install/pipcheck/runtime; **26 passed,2 skipped** |
| Docker-образ из рабочего дерева | coldstart/restart/сохранность данных passed |
| Docker-образ из sdist | build и coldstart/restart/сохранность данных passed |
| Wheel отдельно от checkout |212Python-файлов runtime совпали; bounded rejection passed |
| Ruff |1577существующих замечаний, **новых0** |
| Workflow actionlint,git diff whitespace,self-check нового кода | passed |

В sdist не включены внешние большие benchmark-корпуса: из-за этого17дополнительных
тестов пропущены; в рабочем дереве они прошли. В нативных средах два CPU-теста
пропущены из-за отсутствия необязательного Torch; реальные Torch/BLAS проверки
выполнены в Linux-контейнере. Два тяжёлых latency-теста исключены явно;
результат BGE измерен отдельно и **не прошёл200мс**. Общий lint не объявляется зелёным.

До исправления sdist отсутствовал `tests/conftest.py`: это вызывало ошибку импорта,
а после ручного добавления PYTHONPATH — фоновые обработчики в тестах,
OOM в4GiB Ubuntu VM и SQLite lock на Windows. После восстановления общих
fixtures все26нативных регрессий прошли. Первый полный архивный прогон дополнительно
выявил недостающие support files; они включены, тесты и пороги не ослаблялись.
История неудачных прогонов сохранена рядом с финальными логами.

Предыдущие проверки всех четырёх способов установки, systemd/LaunchAgent/Task
Scheduler и12browser E2E приведены в [platform report](../platform-v14-20260915/RESULTS.md)
и [browser/CPU report](../browser-cpu-v14-20260915/RESULTS.md). Установщики и UI
после тех проверок не менялись. Расширенный GitHub matrix ещё не запускался.

## CPU и качество

Нативная GDB-трассировка выявила дополнительные OpenMP threads, которые не
ограничивались поздней настройкой. Ранний лимит подтвердился реальным loader:
только основной поток расходовал заметное CPU-время. Отношение CPU/ожидания
стало около1 вместо2,9. Но BGE latency выросла: p50 около2132мс,p95 около2179мс
на10коротких парах. Порядок float32 результатов сохранён; max score delta1,4e-9.
INT8 оказался медленнее и изменил порядок, поэтому отклонён.
[Полный CPU-отчёт](../grounded-v14/CPU_RESULTS.md).

Проверены два дополнительных QA-кандидата по200вопросов. Оба дали улучшения
на development gate, но регрессии на повторной validation, поэтому не включены
в runtime. Прежние reader prompts/schema и opt-in статус memory_answer сохранены.
[Prompt-order candidate](../grounded-v14/RESULTS_V4.md),
[answer-first candidate](../answer-first-v14/RESULTS.md).

Дополнительный API-расход этого продолжения с резервами **$1.3127544**;
исторический верхний предел **$22.6955608/$40**, остаток≥**$17.3044392**.
Эталоны использовались только оценщиком, не reader/verifier.

## Артефакты

- `dist/total_agent_memory-14.0.0-py3-none-any.whl`
  SHA256:`ee6a176faa785708fab6517f12e12c11524366e469fa5b5480564689d074494c`
- `dist/total_agent_memory-14.0.0.tar.gz`
  SHA256:`87d08372ad2613e5a502524737960a1c9b4194cc6f58901e242d9d70b6eb3ba6`
- `tam-team-check:14.0.0`
  image:`cddd1a01fcebda4e296ea29a40262d03de4c1715541b5fca672cb60aeb43566e`
- `tam-source-check:14.0.0`
  image:`a7e4e61c60569b8957b0aecb32e2e55e01bd9723193316c141ff0b67a18e740a`

Хеши/сверки: `wheel.json`,`sdist.json`,`verification.json`; логи проверок — рядом.
Тестовые Compose services и transfer server остановлены; Windows VM suspended,
Ubuntu VM stopped. Рабочие сервисы не развёртывались и Git не изменялся.

## Что требуется для публикации

Пользователь проверяет изменения и выполняет commit/push согласно собственному
запрету автоматических Git-операций. Затем должен пройти обязательный CI gate
package/native/compose/browser/source-archive. Публикация и production deployment
являются отдельными действиями; их успех здесь не заявляется.
