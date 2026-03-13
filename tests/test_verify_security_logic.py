import sys

from genericsuite_ai.lib.ai_utilities import is_safe_url, is_safe_local_path


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
