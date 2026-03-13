import os
import sys

from genericsuite_ai.lib.ai_utilities import is_safe_url, is_safe_local_path

# Add current path and genericsuite-be path to sys.path to allow imports
sys.path.append(os.getcwd())
sys.path.append(
    '/Users/carlosramirez/desarrollo/mediabros_repos/github/genericsuite-be')

# Mock log_error to avoid dependency issues if needed,
# though ai_utilities should be importable if its dependencies are present.


def test_verify():
    print(">>> Verifying security helpers...")

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
        print(
            f"  URL: {url} -> {'Safe' if result else 'Unsafe'} " +
            f"(Expected: {'Safe' if expected else 'Unsafe'})")
        assert result == expected

    # Path Tests
    paths = [
        ("/tmp/test.mp3", True),
        ("/etc/passwd", False),
        ("../../etc/passwd", False),
        ("/tmp/../../etc/passwd", False),
    ]

    for path, expected in paths:
        result = is_safe_local_path(path)
        print(
            f"  Path: {path} -> {'Safe' if result else 'Unsafe'} " +
            f"(Expected: {'Safe' if expected else 'Unsafe'})")
        assert result == expected

    print(">>> All security checks verified successfully!")


if __name__ == "__main__":
    try:
        test_verify()
    except Exception as e:
        print(f"!!! Verification failed: {e}")
        sys.exit(1)
