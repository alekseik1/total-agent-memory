# Browser и CPU — кандидат14.0.0,2026-09-15

**DONE_WITH_CONCERNS:** исправление повторных загрузок моделей и браузерная приёмка выполнены. Полный GitHub CI, независимый top-10 и первоначальный расширенный QA-объём не объявляются завершёнными. Публикации нет.

## CPU: причина и результат

Прежний default pool2 обходил personal/team/shared в одном порядке. Каждый следующий поиск вытеснял все ранее прогретые процессы и заново загружал модели. Регрессия воспроизвела проблему до исправления.

Теперь default pool3 сохраняет три области одновременно. При меньшем лимите поиск сначала использует прогретые процессы; окончательная сортировка сохраняет исходный порядок областей при равных score. Тесты проверяют сохранность PID, лимит2 и независимость результата от порядка исполнения.

Один пользователь, три области, по одному реальному источнику в каждой,4разных запроса после записи, FastEmbed MiniLM, один embedding-поток, Docker Linux ARM64. Время CPU учитывает процессы workers, включая завершённые; это не CPU всей машины. Система хоста не выделялась исключительно под benchmark.

| Показатель | До,2workers | После,2workers | После,3workers |
|---|---:|---:|---:|
| CPU-секунды4запросов |17,372597|5,825269|0,17|
| Медиана запроса,мс |3419,16|1165,84|32,73|
| Максимум4запросов,мс |3546,77|1179,96|62,19|
| Сохранившиеся workers после каждого поиска |0|1|3|
| Измеренный прирост worker CPU за5с простоя |0|0,01|0|

Хеш последовательности scope/id/content результатов одинаков во всех вариантах:45b8f13ba5df0508d532e4e220a9483745ca28401ab4c1168677a6118e8f4bb1. Это около99%снижения CPU именно данного микросценария; оно не переносится автоматически на другие базы/пользователей. Нулевой прирост счётчика за5с не означает математически нулевую загрузку.

Цена удержания моделей в RAM измерена отдельно:1315,30MiB суммарного RSS двух workers и1975,47MiB трёх. Shared pages учитываются несколько раз, gateway/ОС не включены. Для трёх прогретых MiniLM рекомендовано минимум4ГиБ серверной RAM с проверкой на собственной базе; лимит1/2 экономит RAM за счёт холодных загрузок.

Артефакты:cpu-before.json,cpu-after-two.json,cpu-after-three.json,cpu-memory-two.json,cpu-memory-three.json. Воспроизведение:tests/smoke/team_cpu_profile.py внутри Docker с PYTHONPATH=/workspace/src и подготовленным FASTEMBED_CACHE_PATH. API-расход:$0.

## Проверки

| Проверка | Результат |
|---|---|
| Полный Python suite,Docker |2155passed,30skipped,160warnings;211,99с |
| Chromium/Firefox/WebKit, финальный runtime wheel |12passed,39,62с |
| Нативная macOS arm64,Python3.12.14 |15регрессий passed,26,32с |
| Нативная Windows11 ARM64 с x64 Python3.12.14 |15регрессий passed,63,36с |
| Нативная Ubuntu24.04 aarch64,Python3.12.3,без Docker |15регрессий passed,25,88с |
| Linux Docker,Python3.10.21 |Wheel/pip check/local+remote runtime passed;16регрессий passed,24,24с |
| Linux Docker,Python3.13.15 |Wheel/pip check/local+remote runtime passed;16регрессий passed,22,62с |
| Production Compose |Cold start, реальные запросы, restart и сохранность истории passed |
| Ruff изменённого runtime/новых тестов,actionlint,diff check |Passed |
| Wheel/sdist build |Passed;220runtime-файлов финального wheel совпали с проверенными исходниками |

Браузеры проверяют вход Васи/Пети, три области, поиск, правку другим автором и историю, ограничения reader, отзыв токена, logout, отсутствие токена в browser storage, безопасный вывод HTML-текста и мобильную ширину. Локальный dashboard: название/версия/дата, поиск, карточка и все вкладки; JS-ошибок и API4xx/5xx в сценарии нет. Сервер, Store и модели настоящие.

Браузерный gate добавлен в .github/workflows/smoke.yml. Команды make browser-image и make test-browser выполняются в Docker. Основная suite пропускает browser module при отсутствии Playwright; отдельный browser gate устанавливает его и запускает все12сценариев. Остальные пропуски не считаются успешными тестами.

Первый дополнительный fixture dashboard не находил top-level module; исправлен путь к файлу установленного пакета. Ubuntu при перезагрузке очистила /tmp, поэтому повторное окружение создано в домашнем каталоге VM. Эти сбои проверки не скрыты как production fixes.

## Готовые артефакты

- Image tam-team-check:14.0.0:sha256:2376fd7bf02a8d91a9a52c3435e95f7f08f79d2f4da23993b0029a855e1d358b.
- Wheel:dbf403ac3a4f95a9bb8f0354cc1e15660d9f6fc2177ae989248fec7eb7386a5c.
- Sdist:e6251b5235e142055fd4c371d4afde7eaa426ec0bcb7e20210df09bc5ddc0864.
- dist/,wheel-source-check.json,candidate-files.json,validation.log,build.log,browser.log,browser-results.xml,compose.log,native-*.log,python-*.log,actionlint.log находятся рядом.
- После браузерного и Python3.10/3.13 прогона изменены README/changelog и пересобраны metadata.220runtime-файлов финального wheel побайтно сверены с проверенным исходником; новая metadata не выдаётся за ещё один browser run.

## До публикации

1. Пользователь фиксирует проверенный кандидат и отправляет его в GitHub. Агент не выполняет commit/push согласно пользовательским AGENTS.md.
2. Запустить release-install-matrix на этом коммите и получить успешный release-platform-gate:package,native,compose,browser. Подготовлены12native-сочетаний ОС/Python; Windows x64 hardware/macOS Intel и полная матрица ещё не проверены.
3. Независимое сопоставимое top-10 сравнение не выполнено. Исторические LoCoMo66,49%/LongMemEval73,20% не становятся новым leaderboard результатом. Провал grounded-reader acceptance, latency тяжёлого BGE и расширение remote catalogue сверх8core-инструментов остаются отдельными открытыми задачами исходного объёма.

Временные контейнеры остановлены, Ubuntu VM остановлена, Windows возвращена в исходное suspended-состояние. Git/production deployment/публикация не выполнялись; рабочая память пользователя не менялась.
