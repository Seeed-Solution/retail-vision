"""cmd/control decision table (BASE-1 §6.7): pure functions + injected
operation callbacks, no sockets. The supervisor (M1.13) provides the ops
object; this module only implements schema/target checks, request-id dedup
(LRU 256, redelivery answers the stored ack with a fresh timestamp but does
not re-execute) and ack construction.
"""
from __future__ import annotations

import logging
import re
from collections import OrderedDict
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger("vision_base.control")

SCHEMA_ID = "vb.command/1"
ACK_SCHEMA_ID = "vb.ack/1"
COMMANDS = ("add_stream", "remove_stream", "set_threshold", "list_streams")
STREAM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Query parameters whose value is a credential (review item 21). Stream URLs
# routinely carry RTSP/HTTP credentials, so an ack that echoes the URL back
# over an unauthenticated MQTT topic hands them to every subscriber.
SENSITIVE_QUERY_KEYS = frozenset({
    "password", "passwd", "pwd", "secret", "token", "auth", "authorization",
    "apikey", "api_key", "access_token", "accesskey", "key", "credential",
})

__all__ = ["CommandError", "ControlPlane", "build_ack", "redact_url"]


class CommandError(Exception):
    """Rejected command; ``str(exc)`` goes into the ack ``error`` field."""


def redact_url(url: str) -> str:
    """Strip credentials from a stream URL before logging or acking it.

    ``rtsp://user:pass@cam/s1?token=abc`` -> ``rtsp://***:***@cam/s1?token=***``.
    """
    if not isinstance(url, str) or not url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<redacted url>"
    netloc = parts.netloc
    if "@" in netloc:
        netloc = "***:***@" + netloc.rsplit("@", 1)[1]
    query = parts.query
    if query:
        # Only the sensitive values are rewritten: re-encoding the whole query
        # would change how every other parameter reads.
        kept = []
        for item in query.split("&"):
            key, sep, _value = item.partition("=")
            kept.append(f"{key}=***" if sep
                        and key.lower() in SENSITIVE_QUERY_KEYS else item)
        query = "&".join(kept)
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def build_ack(now_ms: int, session_id: str, device_id: str, request_id: str,
              command: str, ok: bool, *, error: str = "", applied: dict | None = None) -> dict:
    ack = {"schema": ACK_SCHEMA_ID, "timestamp": now_ms, "session_id": session_id,
           "device_id": device_id, "request_id": request_id, "command": command,
           "ok": ok}
    if not ok:
        ack["error"] = error
    else:
        ack["applied"] = applied if applied is not None else {}
    return ack


def _validate_params(command: str, params) -> None:
    if not isinstance(params, dict):
        raise CommandError("params must be an object")
    if command == "add_stream":
        sid = params.get("stream_id")
        if not sid or not isinstance(sid, str) or not STREAM_ID_RE.match(sid):
            raise CommandError("params.stream_id must match ^[A-Za-z0-9_-]{1,64}$")
        url = params.get("url")
        if not url or not isinstance(url, str):
            raise CommandError("params.url is required")
        if "score_threshold" in params and params["score_threshold"] is not None:
            v = params["score_threshold"]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
                raise CommandError("params.score_threshold must be in [0,1]")
    elif command == "remove_stream":
        sid = params.get("stream_id")
        if not sid or not isinstance(sid, str) or not STREAM_ID_RE.match(sid):
            raise CommandError("params.stream_id must match ^[A-Za-z0-9_-]{1,64}$")
    elif command == "set_threshold":
        sid = params.get("stream_id")
        if not sid or not isinstance(sid, str) or not STREAM_ID_RE.match(sid):
            raise CommandError("params.stream_id must match ^[A-Za-z0-9_-]{1,64}$")
        v = params.get("score_threshold")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
            raise CommandError("params.score_threshold must be in [0,1]")


class ControlPlane:
    def __init__(self, device_id: str, *, seen_size: int = 256):
        self.device_id = device_id
        self._seen: OrderedDict[str, dict] = OrderedDict()
        self._seen_size = seen_size

    def handle(self, req: dict, session_id: str, now_ms: int, ops) -> dict | None:
        """Process one command dict; return the ack dict or None (no ack).

        ``ops`` is an object with ``add_stream(params)``, ``remove_stream(params)``,
        ``set_threshold(params)`` and ``list_streams(params)`` methods; each
        returns the ``applied`` dict or raises CommandError.
        """
        if not isinstance(req, dict):
            return None
        if req.get("schema") != SCHEMA_ID or req.get("device_id") != self.device_id:
            return None
        request_id = req.get("request_id")
        if not request_id or not isinstance(request_id, str):
            # Never log the whole request: add_stream params carry the stream
            # URL, credentials included (review item 21).
            log.warning("cmd/control without request_id "
                        "(device_id=%r command=%r params=%r)",
                        req.get("device_id"), req.get("command"),
                        sorted((req.get("params") or {}).keys())
                        if isinstance(req.get("params"), dict) else None)
            return None
        command = req.get("command")

        # redelivery: answer the stored ack (fresh timestamp), never re-execute
        if request_id in self._seen:
            ack = dict(self._seen[request_id])
            ack["timestamp"] = now_ms
            self._seen.move_to_end(request_id)
            return ack

        try:
            if command not in COMMANDS:
                raise CommandError(f"unknown command: {command!r}")
            params = req.get("params")
            _validate_params(command, params)
            applied = getattr(ops, command)(params)
            ack = build_ack(now_ms, session_id, self.device_id, request_id,
                            command, True, applied=applied)
        except CommandError as exc:
            ack = build_ack(now_ms, session_id, self.device_id, request_id,
                            command, False, error=str(exc))
        except Exception as exc:  # unexpected ops failure must still ack
            ack = build_ack(now_ms, session_id, self.device_id, request_id,
                            command, False, error=f"{type(exc).__name__}: {exc}")

        self._seen[request_id] = ack
        if len(self._seen) > self._seen_size:
            self._seen.popitem(last=False)
        return ack
