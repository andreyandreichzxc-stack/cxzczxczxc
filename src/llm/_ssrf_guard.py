"""SSRF prevention for provider base_url validation.

Uses ipaddress module to catch IPv4-mapped IPv6, hex/octal/decimal IP
representations, link-local, ULA, and other bypass vectors.

Also resolves domain names via DNS and checks resolved IPs against
private/loopback ranges to prevent DNS rebinding attacks.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def _is_ip_blocked(ip_str: str) -> str | None:
    """Check if an IP string is in a blocked range. Returns error reason or None."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return None

    if ip.is_loopback:
        return "loopback"
    if ip.is_private:
        return "private network"
    if ip.is_link_local:
        return "link-local"
    if ip.is_unspecified:
        return "unspecified address"
    # IPv4-mapped IPv6 (::ffff:x.x.x.x)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        mapped = ip.ipv4_mapped
        if (
            mapped.is_loopback
            or mapped.is_private
            or mapped.is_link_local
            or mapped.is_unspecified
        ):
            return "IPv6-mapped private/loopback"
    return None


def validate_base_url(url: str | None) -> str | None:
    """Validate base_url to prevent SSRF to internal networks.

    Raises ValueError if the URL targets a private/loopback/link-local address.
    Also resolves domain names via DNS and checks all resolved IPs.
    Returns the URL unchanged if valid.
    """
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid URL scheme: {parsed.scheme!r} (only http/https allowed)"
        )
    hostname = (parsed.hostname or "").lower()

    # Plain hostname blocklist
    if hostname in ("localhost",):
        raise ValueError("Localhost endpoints not allowed")

    # Try to parse as IP address — catches hex, octal, decimal, IPv6-mapped
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        # hostname is a domain name — resolve via DNS and check all IPs
        try:
            addrinfos = socket.getaddrinfo(
                hostname, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
            )
            for family, _, _, _, sockaddr in addrinfos:
                ip_str = str(sockaddr[0])
                reason = _is_ip_blocked(ip_str)
                if reason:
                    raise ValueError(
                        f"Domain {hostname} resolves to {ip_str} ({reason}). "
                        f"DNS rebinding attack or misconfigured endpoint."
                    )
        except socket.gaierror:
            # DNS resolution failed — domain doesn't exist, let connection fail later
            pass
        return url

    # Direct IP checks
    reason = _is_ip_blocked(hostname)
    if reason:
        raise ValueError(f"Endpoint {hostname} not allowed ({reason})")

    return url
