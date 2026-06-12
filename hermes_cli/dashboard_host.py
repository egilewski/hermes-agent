"""Shared dashboard bind-host classification helpers."""

from __future__ import annotations

import ipaddress
from typing import Optional


_LOOPBACK_HOST_NAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})


def normalize_dashboard_bind_host(host: Optional[str]) -> str:
    """Return a non-empty dashboard bind host string.

    Handles values such as ``None``, ``" [::1] "``, and ``"host."`` by
    defaulting empty input to ``127.0.0.1``, trimming whitespace, removing
    trailing dots, and stripping surrounding IPv6 brackets.
    """
    normalized = str(host or "127.0.0.1").strip().rstrip(".")
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    normalized = normalized.strip().rstrip(".")
    return normalized or "127.0.0.1"


def is_loopback_bind_host(host: Optional[str]) -> bool:
    """Return True when a normalized bind host targets loopback.

    In Python < 3.13, ``IPv6Address.is_loopback`` does not report
    IPv4-mapped IPv6 loopback literals from ``ipaddress.ip_address`` as
    loopback directly. The ``ipv4_mapped`` fallback ensures older runtimes
    classify those binds the same way Python 3.13+ does.
    """
    normalized = normalize_dashboard_bind_host(host).lower()
    if normalized in _LOOPBACK_HOST_NAMES:
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped.is_loopback
    return False
