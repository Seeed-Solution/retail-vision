"""RuntimeClient: spawn/supervise one vb-runtime child process and speak the
§6.3 wire protocol over an AF_UNIX socketpair (spec BASE-1 §6.4).

Threading rules (M1.12): ``request``/``snapshot``/``infer_image`` are thread
safe (socket writes under a lock; waiting for the reply does not hold it).
The reader thread only decodes and dispatches — hooks run on the shard's hook
thread (§6.5.1). Calling ``request`` on the reader thread raises RuntimeError.
"""
from __future__ import annotations

import itertools
import socket
import struct
import subprocess
import threading
from typing import Callable

from . import wire
from .types import Detection, Event, FrameResult, Hello, LetterboxGeom, TensorFrame

__all__ = ["RuntimeError_", "RuntimeGone", "RuntimeClient"]


class RuntimeError_(Exception):
    """Native side answered ok:false."""


class RuntimeGone(Exception):
    """Child process exited / connection EOF."""


_ALIGN = {0: "center", 1: "top_left"}


class _Reply:
    __slots__ = ("event", "data", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.data = None
        self.error: BaseException | None = None


class RuntimeClient:
    def __init__(self, argv: list[str], runtime_cfg_path: str, *,
                 on_frame: Callable[[int, FrameResult], None],
                 on_event: Callable[[int, Event], None],
                 on_stats: Callable[[dict], None],
                 on_state: Callable[[int, str, str], None],
                 on_exit: Callable[[int], None],
                 on_tensors: Callable[[int, TensorFrame], None] | None = None):
        # argv template: occurrences of "{fd}" are replaced by the child's
        # socketpair fd; "--config <runtime_cfg_path>" is appended.
        self.argv = list(argv)
        self.runtime_cfg_path = runtime_cfg_path
        self.on_frame = on_frame
        self.on_event = on_event
        self.on_stats = on_stats
        self.on_state = on_state
        self.on_exit = on_exit
        # Dev mode (§6.12): VBT1 records are accepted only when a tensor
        # callback is registered; otherwise they close the connection.
        self.on_tensors = on_tensors
        self.last_wire_error = ""
        self.proc: subprocess.Popen | None = None
        self.hello: Hello | None = None
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._reader_ident: int | None = None
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[str, _Reply] = {}
        self._req_ids = itertools.count(1)
        self._hello_event = threading.Event()
        self._closed = False

    # ------------------------------------------------------------------ start

    def start(self, hello_timeout_s: float = 10.0) -> Hello:
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        cmd = [a.format(fd=child.fileno()) for a in self.argv]
        cmd += ["--config", self.runtime_cfg_path]
        try:
            self.proc = subprocess.Popen(cmd, pass_fds=(child.fileno(),),
                                         stdin=subprocess.DEVNULL)
        finally:
            child.close()
        self._sock = parent
        self._reader = threading.Thread(target=self._read_loop,
                                        name="vb-runtime-reader", daemon=True)
        self._reader.start()
        if not self._hello_event.wait(hello_timeout_s):
            self.kill()
            raise TimeoutError(f"hello not received within {hello_timeout_s}s")
        if self.hello is None:
            # reader hit EOF before hello: fail fast instead of waiting out the
            # full timeout.
            self.kill()
            raise RuntimeGone("runtime exited before hello")
        return self.hello  # type: ignore[return-value]

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    # --------------------------------------------------------------- requests

    def request(self, op: str, timeout_s: float, **fields) -> dict:
        """Send one control line and block for the matching reply (by req)."""
        if threading.get_ident() == self._reader_ident:
            raise RuntimeError("request from reader thread would deadlock")
        req = f"r-{next(self._req_ids)}"
        msg = {"op": op, "req": req, **fields}
        entry = _Reply()
        with self._pending_lock:
            self._pending[req] = entry
        try:
            line = wire.encode_control_line(msg)
            with self._write_lock:
                if self._sock is None:
                    raise RuntimeGone("runtime closed")
                self._sock.sendall(line)
        except BaseException as e:
            with self._pending_lock:
                self._pending.pop(req, None)
            if isinstance(e, (RuntimeGone, OSError)):
                raise RuntimeGone(f"cannot send {op}: {e}") from e
            raise
        if not entry.event.wait(timeout_s):
            with self._pending_lock:
                self._pending.pop(req, None)
            raise TimeoutError(f"{op} timed out after {timeout_s}s")
        with self._pending_lock:
            self._pending.pop(req, None)
        if entry.error is not None:
            raise entry.error
        if isinstance(entry.data, tuple):        # snapshot (meta, jpeg)
            return entry.data
        if not entry.data.get("ok"):
            raise RuntimeError_(entry.data.get("error") or f"{op} failed")
        return entry.data

    def snapshot(self, stream_index: int, seq: int = 0, track_id: int = 0,
                 crop: bool = True, max_side: int = 640,
                 timeout_s: float = 3.0) -> tuple[dict, bytes]:
        res = self.request("snapshot", timeout_s, stream_index=stream_index,
                           seq=seq, track_id=track_id, crop=crop,
                           max_side=max_side)
        return res  # type: ignore[return-value]

    def infer_image(self, jpeg: bytes, timeout_s: float = 2.0) -> dict:
        import base64
        reply = self.request("infer_image", timeout_s,
                             jpeg_b64=base64.b64encode(jpeg).decode("ascii"))
        return reply.get("applied", {})

    def stop(self, timeout_s: float = 5.0) -> None:
        """Clean stop; SIGTERM after timeout, SIGKILL 2 s later (§6.4)."""
        self._closed = True
        try:
            self.request("stop", timeout_s)
        except (TimeoutError, RuntimeError_, RuntimeGone, RuntimeError):
            pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(2.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(2.0)
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._reader is not None:
            self._reader.join(timeout_s)

    def kill(self) -> None:
        self._closed = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(2.0)
            except subprocess.TimeoutExpired:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._reader is not None:
            self._reader.join(2.0)

    # ------------------------------------------------------------------ reader

    def _read_loop(self) -> None:
        self._reader_ident = threading.get_ident()
        sock = self._sock
        buf = b""
        assert sock is not None
        while True:
            try:
                chunk = sock.recv(65536)
            except OSError:
                chunk = b""
            if not chunk:
                break
            buf += chunk
            while True:
                if len(buf) < wire.HEADER.size:
                    break
                try:
                    magic, n = wire.decode_header(buf[:8])
                except wire.WireError:
                    buf = b""
                    break
                if len(buf) < 8 + n:
                    break
                body = buf[8:8 + n]
                buf = buf[8 + n:]
                try:
                    self._dispatch(magic, body)
                except Exception:
                    pass  # decode errors: drop record, keep the connection
        # EOF: fail all pending waiters, report exit code.
        self._hello_event.set()  # unblock start() if hello never arrived
        with self._pending_lock:
            entries = list(self._pending.values())
            self._pending.clear()
        for e in entries:
            e.error = RuntimeGone("runtime closed connection")
            e.event.set()
        code = self.proc.wait() if self.proc is not None else -1
        # Only post-hello exits go to on_exit; pre-hello failures surface as
        # start() raising and are counted by the caller exactly once.
        if self.hello is None:
            return
        try:
            self.on_exit(code)
        except Exception:
            pass

    def _dispatch(self, magic: bytes, body: bytes) -> None:
        if magic == wire.MAGIC_FRAME:
            self.on_frame(*self._decode_frame(body))
        elif magic == wire.MAGIC_EVENT:
            obj = wire.decode_event(body)
            ev = Event(stream_id="", seq=obj.get("seq", 0),
                       wall_ms=obj.get("wall_ms", 0.0),
                       analyzer=obj.get("analyzer", ""),
                       type=obj.get("type", ""),
                       track_id=obj.get("track_id", 0),
                       fields=obj.get("fields", {}))
            self.on_event(obj.get("stream_index", 0), ev)
        elif magic == wire.MAGIC_CONTROL:
            self._dispatch_control(wire.decode_event(body))
        elif magic == wire.MAGIC_SNAPSHOT:
            meta, jpeg = wire.decode_snapshot(body)
            req = meta.get("req")
            if req:
                with self._pending_lock:
                    entry = self._pending.pop(req, None)
                if entry is not None:
                    entry.data = (meta, jpeg)
                    entry.event.set()
        elif magic == wire.MAGIC_TENSOR:
            self._dispatch_tensor(body)

    def _dispatch_tensor(self, body: bytes) -> None:
        """§6.12: VBT1 outside dev mode = unknown magic -> error + close."""
        if self.on_tensors is None:
            self.last_wire_error = "unexpected VBT1 record outside dev mode"
            try:
                if len(body) < 4:
                    raise wire.WireError("truncated VBT1 body")
                stream_index = struct.unpack_from("<I", body, 0)[0]
            except wire.WireError:
                stream_index = 0
            try:
                self.on_state(stream_index, "error", self.last_wire_error)
            except Exception:
                pass
            sock = self._sock
            self._sock = None
            if sock is not None:
                try:
                    sock.close()      # reader loop exits via EOF/OSError
                except OSError:
                    pass
            return
        tf = wire.decode_tensors(body)
        self.on_tensors(tf.stream_index, tf)   # type: ignore[attr-defined]

    def _dispatch_control(self, obj: dict) -> None:
        op = obj.get("op")
        if op == "hello":
            self.hello = Hello(
                runtime_version=obj.get("runtime_version", ""),
                abi=int(obj.get("abi", 0)),
                backend=obj.get("backend", ""),
                caps=dict(obj.get("caps", {})),
                model_hw=tuple(obj.get("model_hw", (0, 0))),
                model_sha256=obj.get("model_sha256", ""),
                attr_names=tuple(obj.get("attr_names", ())),
                pid=int(obj.get("pid", 0)))
            self._hello_event.set()
        elif op == "reply":
            req = obj.get("req")
            with self._pending_lock:
                entry = self._pending.get(req)
            if entry is not None:
                entry.data = obj
                entry.event.set()
        elif op == "stats":
            self.on_stats(obj)
        elif op == "stream_state":
            self.on_state(int(obj.get("stream_index", 0)),
                          obj.get("state", ""), obj.get("error", "") or "")

    def _decode_frame(self, body: bytes) -> tuple[int, FrameResult]:
        attr_names = self.hello.attr_names if self.hello else ()
        d = wire.decode_result(body, attr_names)
        geom = LetterboxGeom(
            src_w=d["src_w"], src_h=d["src_h"], model_w=d["model_w"],
            model_h=d["model_h"], scale=d["scale"], pad_x=d["pad_x"],
            pad_y=d["pad_y"], align=_ALIGN.get(d["align"], "center"))
        detections = [Detection(
            cx=x["cx"], cy=x["cy"], w=x["w"], h=x["h"], score=x["score"],
            class_id=x["class_id"], track_id=x["track_id"],
            keypoints=tuple(v for kpt in x["keypoints"] for v in kpt),
            attrs=tuple(x["attrs"])) for x in d["detections"]]
        res = FrameResult(stream_id="", seq=d["seq"], wall_ms=d["wall_ms"],
                          geom=geom, detections=detections,
                          inference_ms=d["inference_ms"],
                          queue_delay_ms=d["queue_delay_ms"])
        return d["stream_index"], res
