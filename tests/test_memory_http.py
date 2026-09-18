import io
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from memory_http import JsonHttpClient, MemoryClientError


class MemoryHttpClientTest(unittest.TestCase):
    def test_http_error_includes_bounded_api_detail(self) -> None:
        error = HTTPError(
            "http://127.0.0.1:8001/test",
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(
                '{"detail":"Session başka bir kullanıcıya aittir"}'.encode(
                    "utf-8"
                )
            ),
        )
        with patch("memory_http.request.urlopen", side_effect=error):
            with self.assertRaises(MemoryClientError) as captured:
                JsonHttpClient("http://127.0.0.1:8001", 10).request(
                    "POST", "/test", {"value": "hello"}
                )

        self.assertEqual(
            str(captured.exception),
            "POST /test: HTTP 403: Session başka bir kullanıcıya aittir",
        )

    def test_non_json_http_error_does_not_expose_response_body(self) -> None:
        error = HTTPError(
            "http://127.0.0.1:8001/test",
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b"internal implementation details"),
        )
        with patch("memory_http.request.urlopen", side_effect=error):
            with self.assertRaises(MemoryClientError) as captured:
                JsonHttpClient("http://127.0.0.1:8001", 10).request("GET", "/test")

        self.assertEqual(str(captured.exception), "GET /test: HTTP 500")


if __name__ == "__main__":
    unittest.main()
