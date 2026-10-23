"""Opt-in shared warm TAM daemon.

One streamable-HTTP TAM server per memory directory, bound to loopback and
protected by a bearer token, so every MCP client session reuses one warm
process (one embedding model in RAM, no cold start) instead of spawning its
own. ``tam proxy`` is the stdio front end MCP clients launch; ``tam daemon``
manages the server. See docs/shared-daemon.md.
"""
