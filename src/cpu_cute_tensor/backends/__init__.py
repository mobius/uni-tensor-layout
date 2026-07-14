"""Reference backends (Host numpy first; VE/Phi hooks)."""

from cpu_cute_tensor.backends.host_ref import host_dgemm_reference, host_sharded_dgemm

__all__ = ["host_dgemm_reference", "host_sharded_dgemm"]
