"""Ambient memory: memory that works while the model never calls a memory tool.

Client hooks (hooks/lib/tam_hook.py) call into this package:

* :mod:`ambient.actions` — the agent action log (``agent_actions``, migration 040)
  fed by PostToolUse / PostToolUseFailure and the Codex notify hook;
* :mod:`ambient.prompt_context` — rules and background recalled for every
  prompt (UserPromptSubmit), warm through the daemon, cold through FTS;
* :mod:`ambient.reminders` — a rule reminder before a Bash call or an edit
  whose text matches a rule (PreToolUse);
* :mod:`ambient.turn_end` — the Stop-hook rule compliance check and the turn
  summary auto-save;
* :mod:`ambient.forget` — ``tam forget-auto``.

Modules imported by hooks stay free of the server, the embedder and any
third-party package beyond what :mod:`rule_surfaces` needs.
"""
