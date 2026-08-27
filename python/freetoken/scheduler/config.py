from __future__ import annotations

from dataclasses import dataclass, field

import os

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


def _zmq_addr(slot: int) -> str:
    """Cross-platform ZMQ transport for inter-process links.

    Linux/macOS use ipc:// (fast, no ports). Windows has no reliable ipc://
    transport and no /tmp, so we fall back to loopback TCP on distinct ports.
    """
    if os.name == "nt":
        return f"tcp://127.0.0.1:{5550 + slot}"
    return f"ipc:///tmp/freetoken_{slot}" + _get_pid_suffix()


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return _zmq_addr(0)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return _zmq_addr(1)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return _zmq_addr(2)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
