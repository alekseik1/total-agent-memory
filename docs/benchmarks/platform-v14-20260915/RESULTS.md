# Проверки платформ 14.0.0 — 2026-09-15

**DONE_WITH_CONCERNS:** все четыре запрошенных способа запуска прошли в перечисленных окружениях. Это подтверждает установку и основные рабочие сценарии; весь первоначальный объём релиза и все сочетания ОС/архитектур/Python этим не закрываются.

## Итоговые проверки

| Окружение | Результат |
|---|---|
| Linux Docker Compose, aarch64, Python3.12.14 | Production image: cold start, реальный remote MCP, запись, авторство, отзыв доступа, stop/start, сохранность записей и истории — passed |
| Ubuntu24.04 без Docker, aarch64, Python3.12.3 | Отдельная VZ VM без Docker/containerd. Wheel/runtime,13регрессий (9,14с), полный install.sh, source runtime, systemd dashboard и HTTP /api/release — passed |
| macOS arm64, Python3.12.14 | Wheel/runtime,13регрессий (11,07с), полный install.sh, source runtime, настоящий временный LaunchAgent и HTTP /api/release — passed |
| Windows11 ARM64, CPython3.12.14 x64 через эмуляцию, PowerShell5.1 | Wheel/runtime и20регрессий (68,25с) — passed. После исправлений установщика: полный install.ps1, source runtime, Task Scheduler dashboard и HTTP /api/release — passed;8тестов установщика — passed (24,50с) |
| Полный Python suite после последних изменений, Docker | **2152 passed,29 skipped,160 warnings;190,25с**. Восемь PowerShell-пропусков выполнены отдельно на Windows |
| Ruff нового серверного пакета и изменённых проверок | Passed |
| Wheel + sdist build | Passed. Содержимое всех файлов финального wheel побайтно совпало с wheel, проверенным нативно; ZIP timestamps могут отличаться |
| actionlint для release-install-matrix | Passed |

Сквозной сценарий проверяет personal/team/shared, кириллицу, принадлежность автора токену, обновление/историю, request_id, отзыв членства, перезапуск, offline backup/restore и локальный stdio MCP. Проверка dashboard получает фактические название, версию и дату через /api/release.

## Исправления, найденные нативными тестами

- SQL-миграции, словарь тегов, JSONL и экспорт явно используют UTF-8. Windows cp1252 больше не ломает инициализацию хранилища и кириллицу. Регрессия воспроизводит legacy-кодировку на любой ОС.
- ServerLease сначала захватывает блокировку, затем читает/инициализирует байт. На Windows чтение уже заблокированного байта прежде выпадало как необработанный PermissionError.
- Windows installer обновляет pip через python -m pip: запуск pip.exe для собственного обновления срывал полную установку.
- Все фоновые Windows-задачи получают выбранную папку памяти через UTF-8 Python launcher; сохраняются аргументы, пробелы и кириллица.
- Dashboard AtLogon привязан к текущей Windows identity. Общий триггер давал Access denied обычному пользователю. Ошибка регистрации теперь выводится с исходной причиной. Параметр User описан в [Microsoft Learn](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasktrigger).
- Smoke-клиент ждёт ответ initialize, затем tools/call и только потом закрывает stdin. Пакетный ввод с немедленным EOF на Ubuntu отменял незавершённый ответ SDK. Это было исправление проверки, а не production MCP.
- Windows smoke запускает сервер через Python entrypoint и завершает CTRL_BREAK_EVENT: остановка console .exe launcher раньше оставляла дочерний Python с занятой блокировкой.

## Границы доказанного

- CI подготовлен на Ubuntu24.04, Windows2022, macOS15 Intel/ARM × Python3.10/3.12/3.13 плюс Compose. Эти12CI-сочетаний не запускались: проверен локальный кандидат, Git остаётся read-only. Windows проверялась на ARM64 с x64 Python; Windows x64 hardware, macOS Intel и остальные Python требуют CI.
- Полный репозиторный lint имеет прежние замечания (предыдущий аудит1578); чистым объявляется только указанный проверенный код.29пропусков full suite не приравниваются к успешным тестам.
- Независимый top-10, исправление провала grounded-reader acceptance, бюджет latency тяжёлого BGE, браузерный E2E и расширение удалённого каталога сверх8core-инструментов остаются открытыми задачами исходного релиза.
- В Windows certificate store есть повреждённый сертификат, Python выводит предупреждение. TLS не отключался. Установщик также ошибочно трактует warning модели на stderr как повод сообщить о загрузке при первом использовании; модель фактически загрузилась, реальная запись и поиск прошли.
- Production deployment, публикация пакета, Git commit/push не выполнялись.

## Артефакты

- Image tam-team-check:14.0.0: sha256:dfa48933d27042353f91f7dd37b9be975721d802088dfc773095bb0ef45c20c4.
- Wheel: 9e4bb2e79a17fd34efec2175ba53d556ea1c117cc5a9fe3c31b15edab9314b94.
- Sdist: d6186e12f5bb3f23d420cd6e6f38a744749b469665b50cb84667fb1d12c20777.
- native-linux.log, native-macos.log, native-windows-regressions.log, native-windows-source.log — реальные нативные логи. Windows source runtime/8pytest также записаны в native-windows-final.log.
- compose-runtime.log, full-suite.log, final-validation.log, actionlint.log — финальные контейнерные проверки.
- candidate-files.json — SHA256 ключевых файлов кандидата; dist/ — wheel/sdist.
- Временная Ubuntu VM остановлена, macOS LaunchAgent выгружена, Compose test stack и transfer HTTP server остановлены. Четыре Windows-задачи удалены после проверки точного пути к временному Python; Windows VM возвращена в исходное suspended-состояние. Тестовые каталоги/диски сохранены для воспроизведения.
