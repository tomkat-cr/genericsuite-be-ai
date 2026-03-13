import pytest

from genericsuite_ai.lib.ai_utilities import is_safe_url, is_safe_local_path

from genericsuite_ai.lib.ai_vision import \
    vision_image_analyzer

from genericsuite_ai.lib.ai_audio_processing import \
    process_audio_url, process_audio_file


def test_is_safe_url():
    # Safe URLs
    assert is_safe_url("https://www.google.com") is True
    assert is_safe_url("http://example.com") is True
    # Restricted URLs (IPs)
    assert is_safe_url("http://127.0.0.1") is False
    assert is_safe_url("http://169.254.169.254") is False
    assert is_safe_url("http://10.0.0.1") is False
    assert is_safe_url("http://192.168.1.1") is False
    # Restricted Schemes
    assert is_safe_url("file:///etc/passwd") is False
    assert is_safe_url("ftp://example.com") is False
    # Restricted Hostnames
    assert is_safe_url("http://localhost") is False


def test_is_safe_local_path():
    # Safe paths
    assert is_safe_local_path("/tmp/test.mp3") is True
    # Unsafe paths
    assert is_safe_local_path("/etc/passwd") is False
    assert is_safe_local_path("/var/log/syslog") is False
    # Path traversal
    assert is_safe_local_path("/tmp/../../etc/passwd") is False
    assert is_safe_local_path("../../etc/passwd") is False


def test_audio_processing_lfi_protection():
    # Test process_audio_file with unsafe local path
    with pytest.raises(Exception) as excinfo:
        process_audio_file("/etc/passwd", lambda **x: x, {})
    assert "Unsafe local path" in str(excinfo.value)


def test_audio_processing_ssrf_protection():
    # Test process_audio_file with unsafe URL
    with pytest.raises(Exception) as excinfo:
        process_audio_file("http://127.0.0.1/test.mp3", lambda **x: x, {})
    assert "Unsafe URL" in str(excinfo.value)


def test_audio_processing_url_resultset_protection():
    # Test process_audio_url with unsafe URL
    result = process_audio_url(
        "http://127.0.0.1/test.mp3", lambda **x: x, {}, "url")
    assert result["error"] is True
    assert "Unsafe URL" in result["error_message"]

    # Test process_audio_url with unsafe local path
    result = process_audio_url("/etc/passwd", lambda **x: x, {}, "url")
    assert result["error"] is True
    assert "Unsafe local path" in result["error_message"]


def test_vision_image_analyzer_protection():
    # Test vision_image_analyzer with unsafe local path
    params = {"image_path": "/etc/passwd", "question": "What is this?"}
    result = vision_image_analyzer(params)
    assert result["error"] is True
    assert "Unsafe local path" in result["error_message"]

    # Test vision_image_analyzer with unsafe URL
    params = {"image_path": "http://127.0.0.1/test.jpg",
              "question": "What is this?"}
    result = vision_image_analyzer(params)
    assert result["error"] is True
    assert "Unsafe URL" in result["error_message"]
