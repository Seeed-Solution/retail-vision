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

log = logging.getLogger("vision_base.control")

SCHEMA_ID = "vb.command/1"
ACK_SCHEMA_ID = "vb.ack/1"
COMMANDS = ("add_stream", "remove_stream", "set_threshold", "list_streams")
STREAM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

__all__ = ["CommandError", "ControlPlane", "build_ack"]


class CommandError(Exception):
    """Rejected command; ``str(exc)`` goes into the ack ``error`` field."""


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
            log.warning("cmd/control without request_id: %r", req)
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
