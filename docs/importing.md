# Importing history and moving the store

## `tam import`: chat exports, mem0 and Markdown notes

```bash
tam import chatgpt  ~/Downloads/chatgpt-export/conversations.json --project personal
tam import claude   ~/Downloads/claude-export/conversations.json  --project personal --include-assistant
tam import mem0     mem0-memories.json                            --project assistant
tam import markdown ~/Notes                                       --project notes --dry-run
```

| Kind | Input | One record per |
|---|---|---|
| `chatgpt` | `conversations.json` from ChatGPT → Settings → Data controls → Export | message on the visible branch of each conversation (edited-away branches are skipped) |
| `claude` | `conversations.json` from claude.ai → Settings → Privacy → Export data | message in `chat_messages` |
| `mem0` | JSON of `Memory.get_all()` / `MemoryClient.get_all()`: `{"results": [...]}` (or `"memories"`), or a bare list | memory (`memory` field) |
| `markdown` | a directory (or one `.md` file) | note, or heading section (`#`–`###`) when the note body is over 4,000 characters |

Options:

- `--project P`: project to store into (default `general`).
- `--dry-run`: parse and count without writing anything. It still reads the import ledger, so it reports what an import would skip.
- `--include-assistant`: chat exports import user turns only by default. This flag adds the assistant replies. System, tool and hidden messages are never imported.
- `--memory-dir D`: store location (default `TAM_MEMORY_DIR` or `~/.tam`).
- `--quiet`: no progress lines on stderr.

The summary is printed as JSON on stdout: `units_seen`, `imported`, `deduplicated` (an equal record was already stored), `already_imported` (found in the import ledger), `skipped_role`, `skipped_empty`, `skipped_short` (turns under 12 characters, such as "ok" or "thanks"), `skipped_files`, `redacted`, `split_parts`, and the time range found.

How records are stored:

- **The normal save path.** Every unit goes through `save_knowledge`: `<private>` stripping, credential redaction, duplicate detection, embedding and graph linking. It is saved as a `fact` tagged `imported`, `import-<kind>` and the source tags: `chatgpt` / `claude-ai` with `role-user` / `role-assistant`, mem0 `categories`, Markdown frontmatter `tags`. The context line names the conversation or note. The LLM quality gate is skipped, because one LLM verdict per turn would make a bulk import slow and, with a paid provider, costly.
- **Original timestamps.** `created_at` and `last_confirmed` are set to the message time, the mem0 `created_at`, or for Markdown the frontmatter `created` / `date` / `created_at` with the file modification time as fallback. Timelines, reports and date-filtered recall see the history at the time it happened. An imported duplicate of a record you already have never back-dates that record.
- **Idempotent.** The `import_ledger` table (migration 042) keeps a SHA-256 of kind, project, exact text and original time for every stored unit. Running the same import again skips them. An edited note is new text, so it is stored again and the old version stays.
- **Large inputs.** Exports are parsed as a stream, one conversation at a time, so a multi-gigabyte `conversations.json` is not loaded whole. Messages over 6,000 characters are split on paragraph boundaries into `(part N/M)` records. Markdown files over 5 MB, hidden directories (`.obsidian`, `.git`, `.trash`) and symlinks are skipped.
- **Strict format checks.** Feeding a claude.ai file to the `chatgpt` importer, or a mem0 file without `memory` fields, stops with a message saying what was expected (exit code 2). Platform exports built from a custom `create_memory_export` schema have a user-defined shape: convert them to the `get_all()` list first.

`scripts/cm-import` is the old author-specific importer. It reads fixed paths and writes the database directly. Use `tam import markdown` instead.

## `tam export --encrypted` / `tam import-bundle`: move the store between machines

```bash
export TAM_BUNDLE_PASSPHRASE='a long passphrase'        # or --key-file, or the interactive prompt
tam export --encrypted ~/memory-2026-10.tambundle            # the whole store
tam export --encrypted ~/billing.tambundle --project billing # one project's active records

# on the other machine
tam import-bundle ~/memory-2026-10.tambundle --mode replace  # restore (stop TAM servers first)
tam import-bundle ~/billing.tambundle                        # merge (default)
```

**Export** snapshots `memory.db` with SQLite's online backup API, so data still in the WAL is included and a running server is not interrupted. The embeddings table, which is TAM's vector store, travels inside the snapshot. The Chroma directory is a derived mirror whose files cannot be copied consistently, so it is not bundled and is rebuilt on import. `--project` exports that project's active records as JSON lines instead, and such a bundle can only be merged. The bundle file is written atomically with mode 0600 and is not overwritten without `--force`.

**Format.** A gzip'd tar of `manifest.json` (SHA-256 and size of every member, kind, schema and TAM version) plus `memory.db` or `knowledge.jsonl`. It is sealed with AES-256-GCM in 1 MiB authenticated chunks, and the key comes from the passphrase through scrypt (n=2^17, r=8, p=1, random salt). The versioned header is bound to every chunk as associated data, and the last chunk is marked, so a changed header, a flipped byte, reordered chunks or a truncated file are all detected. Export requires a passphrase of at least 12 characters. Passphrase sources, in order: `--key-file` (a warning is printed if other users can read it), the variable named by `--passphrase-env` (default `TAM_BUNDLE_PASSPHRASE`), then an interactive prompt.

**Import** decrypts into a private staging directory under `backups/` and checks the bundle before the live store is touched:

- every tar member must be one of `manifest.json`, `memory.db`, `knowledge.jsonl` and a regular file. Paths with directories or `..`, absolute paths, symlinks, hard links, devices and duplicates are refused. Members are copied out by name, never extracted by `tarfile`, and nothing is unpickled;
- sizes and SHA-256 must match the manifest;
- the database must pass `PRAGMA integrity_check`, contain TAM's tables, and not come from a newer schema than this TAM ships.

Then, by mode:

- `--mode merge` (default): every active record goes through the normal save path with its type, tags, context and original `created_at`. Duplicates are detected and the import ledger makes repeated merges no-ops.
- `--mode replace`: the current store is first copied to `backups/pre-import-<time>.db`, and the old Chroma directory is moved to `backups/pre-import-<time>-chroma`. Then the snapshot is restored through the backup API, which is safe with WAL, and the vector revision is bumped. Newer migrations are applied on open. Vectors made by a different embedding model than the one configured here are dropped and re-embedded through the `memory_rebuild_embeddings` path. The Chroma mirror is refilled from the embeddings table when Chroma is in use. To undo, restore the safety backup the same way.

Wrong passphrase, tampering, a malformed archive or a failed check exits with code 2 and leaves the live store unchanged. The plain `memory_export` MCP tool is unchanged and still writes unencrypted JSON to `backups/`.
