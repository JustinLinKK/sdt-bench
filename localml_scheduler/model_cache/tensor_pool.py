"""TensorPoolManager: cross-process baseline weight sharing via CUDA IPC.

Background:
  When N concurrent jobs use the same baseline (e.g. R50 ImageNet weights),
  the default behavior is each worker calls torch.load on the baseline file
  and pushes its own copy to GPU. With 4 jobs * 100 MB R50 = 400 MB wasted.

Approach:
  A small background server process owns the master GPU tensors and
  serializes IPC handles via a unix-domain socket. Worker subprocesses
  request a baseline by id; the server returns a tuple
  (rebuild_fn_qualname, rebuild_args_pickled) which the worker passes to
  `torch.multiprocessing.reductions.rebuild_cuda_tensor` to attach.

  All workers see the SAME GPU memory. Each worker is expected to detach()
  + clone() before mutating (gradients), so weights remain shared, and only
  per-job mutations cost extra VRAM.

Caveats:
  - The producer (server) MUST stay alive until all consumers release.
    `torch.multiprocessing` warns "Producer process has been terminated"
    if the producer dies first; rebuild may then segfault.
  - On first attach the consumer's CUDA context is bound to the same device.
    This is the same device the producer used.
  - cuIpcCloseMemHandle is implicit; we just drop refs.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import socket
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("localml_scheduler")


# Wire format: 4-byte big-endian length prefix + pickled python object.
_LEN_FMT = ">I"


def _send_msg(conn: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload)
    conn.sendall(struct.pack(_LEN_FMT, len(data)))
    conn.sendall(data)


def _recv_msg(conn: socket.socket) -> Any:
    hdr = b""
    while len(hdr) < 4:
        chunk = conn.recv(4 - len(hdr))
        if not chunk:
            raise EOFError("connection closed during header read")
        hdr += chunk
    (n,) = struct.unpack(_LEN_FMT, hdr)
    body = b""
    while len(body) < n:
        chunk = conn.recv(min(65536, n - len(body)))
        if not chunk:
            raise EOFError("connection closed during body read")
        body += chunk
    return pickle.loads(body)


@dataclass(slots=True)
class _PoolEntry:
    baseline_id: str
    tensor: Any  # torch.Tensor on GPU (kept alive by this entry)
    rebuild_fn_qualname: str
    rebuild_args: bytes  # pickled
    refcount: int = 0
    bytes_on_gpu: int = 0
    notes: dict = field(default_factory=dict)


class TensorPoolServer:
    """Background server holding master GPU tensors. Run as a subprocess of
    the scheduler service.

    Wire commands (request / reply):
      REGISTER {baseline_id, baseline_path, loader_target?} -> {ok, bytes_on_gpu}
      ATTACH   {baseline_id}                                 -> {ok, rebuild_fn, rebuild_args, bytes_on_gpu}
      RELEASE  {baseline_id}                                 -> {ok}
      STATS    {}                                             -> {entries, total_bytes_gpu, refcounts}
      SHUTDOWN {}                                             -> {ok}; server exits
    """

    def __init__(self, socket_path: Path, device_id: int = 0) -> None:
        self.socket_path = Path(socket_path)
        self.device_id = device_id
        self._entries: dict[str, _PoolEntry] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock: socket.socket | None = None

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self.socket_path))
        self._sock.listen(64)
        logger.info("[tensor_pool] server listening at %s device=%d", self.socket_path, self.device_id)
        # Pre-init CUDA so subsequent IPC handles are valid.
        try:
            import torch
            torch.cuda.set_device(self.device_id)
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning("[tensor_pool] cuda init failed: %s", exc)

        try:
            while not self._stop.is_set():
                self._sock.settimeout(0.5)
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            self._sock.close()
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def _handle(self, conn: socket.socket) -> None:
        try:
            while True:
                try:
                    req = _recv_msg(conn)
                except EOFError:
                    return
                cmd = req.get("cmd")
                try:
                    if cmd == "REGISTER":
                        reply = self._handle_register(req)
                    elif cmd == "ATTACH":
                        reply = self._handle_attach(req)
                    elif cmd == "RELEASE":
                        reply = self._handle_release(req)
                    elif cmd == "STATS":
                        reply = self._handle_stats()
                    elif cmd == "SHUTDOWN":
                        self._stop.set()
                        reply = {"ok": True}
                    else:
                        reply = {"ok": False, "error": f"unknown cmd {cmd}"}
                except Exception as exc:
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                _send_msg(conn, reply)
        finally:
            conn.close()

    def _handle_register(self, req: dict) -> dict:
        bid = req["baseline_id"]
        path = req["baseline_path"]
        with self._lock:
            if bid in self._entries:
                e = self._entries[bid]
                return {"ok": True, "bytes_on_gpu": e.bytes_on_gpu, "already": True}
        # Load checkpoint and place each tensor on GPU. Each tensor gets its
        # own IPC handle (rebuild_args). This avoids the dtype-view mess that
        # comes with packing a heterogeneous state_dict into a flat byte buffer.
        import torch
        from torch.multiprocessing.reductions import reduce_tensor
        obj = torch.load(path, map_location=f"cuda:{self.device_id}", weights_only=False)
        if isinstance(obj, torch.Tensor):
            payload = {"_tensor": obj.to(f"cuda:{self.device_id}")}
        elif isinstance(obj, dict):
            payload = {}
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    payload[k] = v.to(f"cuda:{self.device_id}")
            if not payload:
                return {"ok": False, "error": "no tensors in checkpoint"}
        else:
            return {"ok": False, "error": f"unsupported baseline payload: {type(obj).__name__}"}

        rebuild_qualname = None
        per_key_rebuild_args: dict[str, tuple] = {}  # raw tuples; pickled together below
        per_key_meta: list[dict] = []
        bytes_on_gpu = 0
        for k, v in payload.items():
            if v.numel() == 0:
                per_key_meta.append({"key": k, "shape": tuple(v.shape), "dtype": str(v.dtype),
                                     "numel": 0, "empty": True})
                continue
            rebuild_fn, rebuild_args = reduce_tensor(v)
            qualname = f"{rebuild_fn.__module__}:{rebuild_fn.__name__}"
            if rebuild_qualname is None:
                rebuild_qualname = qualname
            elif rebuild_qualname != qualname:
                logger.warning("[tensor_pool] heterogeneous rebuild fn for key %s", k)
            per_key_rebuild_args[k] = rebuild_args
            per_key_meta.append({"key": k, "shape": tuple(v.shape), "dtype": str(v.dtype),
                                 "numel": v.numel(), "empty": False,
                                 "bytes": v.numel() * v.element_size()})
            bytes_on_gpu += v.numel() * v.element_size()

        with self._lock:
            self._entries[bid] = _PoolEntry(
                baseline_id=bid,
                tensor=payload,  # keep dict alive so GPU mem stays valid for IPC
                rebuild_fn_qualname=rebuild_qualname or "torch.multiprocessing.reductions:rebuild_cuda_tensor",
                rebuild_args=pickle.dumps(per_key_rebuild_args),
                refcount=0,
                bytes_on_gpu=bytes_on_gpu,
                notes={"per_key_meta": per_key_meta},
            )
        return {"ok": True, "bytes_on_gpu": bytes_on_gpu, "n_tensors": len(payload)}

    def _handle_attach(self, req: dict) -> dict:
        bid = req["baseline_id"]
        with self._lock:
            e = self._entries.get(bid)
            if e is None:
                return {"ok": False, "error": f"baseline {bid} not registered"}
            e.refcount += 1
            return {
                "ok": True,
                "rebuild_fn_qualname": e.rebuild_fn_qualname,
                "per_key_rebuild_args": e.rebuild_args,
                "bytes_on_gpu": e.bytes_on_gpu,
                "per_key_meta": e.notes.get("per_key_meta"),
                "refcount": e.refcount,
            }

    def _handle_release(self, req: dict) -> dict:
        bid = req["baseline_id"]
        with self._lock:
            e = self._entries.get(bid)
            if e is None:
                return {"ok": True, "noop": True}
            e.refcount = max(0, e.refcount - 1)
            return {"ok": True, "refcount": e.refcount}

    def _handle_stats(self) -> dict:
        with self._lock:
            return {
                "ok": True,
                "entries": [
                    {"baseline_id": e.baseline_id, "bytes_on_gpu": e.bytes_on_gpu, "refcount": e.refcount}
                    for e in self._entries.values()
                ],
                "total_bytes_gpu": sum(e.bytes_on_gpu for e in self._entries.values()),
            }


class TensorPoolClient:
    """Thin client for workers / scheduler to talk to TensorPoolServer."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = Path(socket_path)

    def _request(self, payload: dict) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(str(self.socket_path))
        try:
            _send_msg(s, payload)
            return _recv_msg(s)
        finally:
            s.close()

    def register(self, baseline_id: str, baseline_path: str) -> dict:
        return self._request({"cmd": "REGISTER", "baseline_id": baseline_id, "baseline_path": baseline_path})

    def attach(self, baseline_id: str) -> dict:
        return self._request({"cmd": "ATTACH", "baseline_id": baseline_id})

    def release(self, baseline_id: str) -> dict:
        return self._request({"cmd": "RELEASE", "baseline_id": baseline_id})

    def stats(self) -> dict:
        return self._request({"cmd": "STATS"})

    def shutdown(self) -> dict:
        return self._request({"cmd": "SHUTDOWN"})


def attach_in_worker(socket_path: str | Path, baseline_id: str) -> tuple[dict, dict, int]:
    """Worker-side helper: connect, attach, rebuild every tensor, return state_dict.

    Returns (state_dict, per_key_meta, bytes_on_gpu).
    """
    import importlib
    import torch  # noqa: F401
    client = TensorPoolClient(socket_path)
    reply = client.attach(baseline_id)
    if not reply.get("ok"):
        raise RuntimeError(f"attach failed: {reply.get('error')}")
    qualname = reply["rebuild_fn_qualname"]
    mod_name, fn_name = qualname.split(":", 1)
    mod = importlib.import_module(mod_name)
    rebuild_fn = getattr(mod, fn_name)
    per_key_rebuild_args = pickle.loads(reply["per_key_rebuild_args"])
    per_key_meta = reply.get("per_key_meta") or []

    state_dict: dict = {}
    for m in per_key_meta:
        k = m["key"]
        if m.get("empty"):
            import torch
            dtype = _dtype_from_str(m["dtype"])
            state_dict[k] = torch.empty(m["shape"], dtype=dtype)
            continue
        args = per_key_rebuild_args.get(k)
        if args is None:
            continue
        state_dict[k] = rebuild_fn(*args)
    return state_dict, per_key_meta, int(reply.get("bytes_on_gpu") or 0)


def _dtype_from_str(s: str):
    import torch
    table = {
        "torch.float32": torch.float32, "torch.float64": torch.float64,
        "torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64, "torch.int32": torch.int32,
        "torch.uint8": torch.uint8, "torch.int8": torch.int8,
        "torch.bool": torch.bool,
    }
    return table.get(s, torch.float32)
