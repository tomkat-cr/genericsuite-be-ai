import os
import sys
import ipaddress
import socket
from urllib.parse import urlparse

# Redefining the logic from ai_utilities.py for standalone verification


def is_safe_url(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False
        hostname = parsed.hostname
        if not hostname:
            return False

        # Resolve hostname to IP
        ip_addr = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_addr)
        # print(f"DEBUG: {hostname} resolved to {ip_addr}")

        if ip.is_loopback or ip.is_private or ip.is_link_local or \
           ip.is_multicast or ip.is_unspecified:
            print(f"DEBUG: Restricted IP: {ip}")
            return False
        return True
    except (ValueError, socket.gaierror) as e:
        print(f"DEBUG: Error resolving {hostname}: {e}")
        return False


def is_safe_local_path(path: str, allowed_dirs=None) -> bool:
    if not path:
        return False

    # Default allowed directories: /tmp and current working directory
    if allowed_dirs is None:
        allowed_dirs = ["/tmp", os.getcwd()]

    try:
        resolved_path = os.path.realpath(path)
        for allowed_dir in allowed_dirs:
            resolved_allowed = os.path.realpath(allowed_dir)
            if resolved_path.startswith(resolved_allowed):
                return True
    except Exception:
        pass

    return False


def test_verify():
    print(">>> Verifying standalone security logic...")

    # URL Tests
    urls = [
        ("https://www.google.com", True),
        ("http://example.com", True),
        ("http://127.0.0.1", False),
        ("http://localhost", False),
        ("file:///etc/passwd", False),
        ("http://169.254.169.254", False),
        ("http://10.0.0.1", False),
    ]

    for url, expected in urls:
        result = is_safe_url(url)
        status = 'Safe' if result else 'Unsafe'
        print(
            f"  URL: {url} -> {status} (Expected: " +
            f"{'Safe' if expected else 'Unsafe'})")
        assert result == expected

    # Path Tests
    # Note: realpath resolution depends on the environment
    paths = [
        ("/tmp/test.mp3", True),
        ("/etc/passwd", False),
        ("../../etc/passwd", False),
    ]

    for path, expected in paths:
        result = is_safe_local_path(path)
        status = 'Safe' if result else 'Unsafe'
        print(
            f"  Path: {path} -> {status} (Expected: " +
            f"{'Safe' if expected else 'Unsafe'})")
        # In some environments /etc/passwd might be in a different place
        # or link, but usually it's outside /tmp and os.getcwd().
        # We assume it should be unsafe here.
        assert result == expected

    print(">>> All security checks verified successfully!")


if __name__ == "__main__":
    try:
        test_verify()
    except AssertionError as e:
        print(f"!!! Verification failed: Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"!!! Verification failed: {e}")
        sys.exit(1)
