"""Exercise real HTTP traffic over a local SOCKS5 endpoint, without Google/WARP."""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import socketserver
import ssl
import struct
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from gemini_web2api import gemini, multimodal, network
from gemini_web2api.config import CONFIG, DEFAULT_CONFIG


class SocksEndpoint(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(5)
        version, count = self.rfile.read(2)
        assert version == 5
        self.rfile.read(count)
        self.wfile.write(b"\x05\x00")
        self.wfile.flush()
        version, command, reserved, address_type = self.rfile.read(4)
        assert (version, command, address_type) == (5, 1, 3)
        host = self.rfile.read(self.rfile.read(1)[0]).decode()
        port = struct.unpack("!H", self.rfile.read(2))[0]
        self.server.destinations.append((host, port))
        self.wfile.write(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x00")
        self.wfile.flush()
        method, path, protocol = self.rfile.readline().decode().strip().split()
        headers = {}
        while True:
            line = self.rfile.readline().decode().strip()
            if not line:
                break
            key, value = line.split(":", 1)
            headers[key.lower()] = value.strip()
        body = self.rfile.read(int(headers.get("content-length", 0)))
        self.server.requests.append((method, path, headers, body))
        status = "200 OK"
        extra_headers = ""
        if path == "/generate":
            inner = [None, None, None, None, [[None, ["hello"], "x" * 250]]]
            response = (json.dumps([["wrb.fr", None, json.dumps(inner)]]) + "\n").encode()
        elif path == "/error":
            status, response = "400 Bad Request", b"location unsupported"
        elif path == "/start":
            extra_headers = "X-Goog-Upload-URL: https://upload.invalid/finalize\r\n"
            response = b""
        elif path == "/finalize":
            response = b"/uploaded/test-image"
        elif path == "/page":
            response = b'{"qKIAYe":"push","Ylro7b":"pctx"}'
        else:
            response = b"image bytes"
        self.wfile.write((
            "HTTP/1.1 {}\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n".format(
                status, len(response), extra_headers,
            )
        ).encode() + response)
        self.wfile.flush()


class SocksServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class WarpRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proxy = SocksServer(("127.0.0.1", 0), SocksEndpoint)
        cls.proxy.destinations = []
        cls.proxy.requests = []
        cls.thread = threading.Thread(target=cls.proxy.serve_forever, daemon=True)
        cls.thread.start()
        spec = importlib.util.spec_from_file_location(
            "legacy_gemini", Path(__file__).resolve().parents[1] / "gemini_web2api.py",
        )
        cls.legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.legacy)

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.saved_config = dict(CONFIG)
        self.saved_legacy_config = dict(self.legacy.CONFIG)
        CONFIG.update(DEFAULT_CONFIG)
        CONFIG.update(
            warp_enabled=True,
            warp_proxy="socks5://127.0.0.1:{}".format(self.proxy.server_address[1]),
            retry_attempts=1, log_requests=False,
        )
        self.legacy.CONFIG.update(CONFIG)
        self.proxy.destinations.clear()
        self.proxy.requests.clear()
        # A conflicting environment proxy and NO_PROXY must not bypass WARP.
        self.env = mock.patch.dict(os.environ, {
            "HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1",
            "NO_PROXY": "*", "GEMINI_WARP_ENABLED": "true",
            "GEMINI_WARP_PROXY": CONFIG["warp_proxy"],
        })
        self.env.start()

    def tearDown(self):
        if gemini._httpx_client is not None:
            gemini._httpx_client.close()
            gemini._httpx_client = None
        CONFIG.clear()
        CONFIG.update(self.saved_config)
        self.legacy.CONFIG.clear()
        self.legacy.CONFIG.update(self.saved_legacy_config)
        self.env.stop()

    def request(self, path):
        return network.open_request(
            urllib.request.Request("http://upstream.invalid:80" + path), CONFIG,
            timeout=5, context=ssl.create_default_context(),
        )

    def test_socks_uses_remote_dns_despite_environment(self):
        with self.request("/image") as response:
            self.assertEqual(response.read(), b"image bytes")
        self.assertEqual(self.proxy.destinations, [("upstream.invalid", 80)])

    def test_socks_http_errors_preserve_status_and_body(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/error")
        with caught.exception as error:
            self.assertEqual(error.code, 400)
            self.assertEqual(error.read(), b"location unsupported")

    def test_generation_streaming_and_nonstreaming_use_warp(self):
        url = "http://upstream.invalid/generate"
        with mock.patch.object(gemini, "_get_url", return_value=url):
            self.assertEqual(gemini.generate("hello", 1, 4), "hello")
            self.assertEqual("".join(gemini.generate_stream("hello", 1, 4)), "hello")
        self.assertEqual(len(self.proxy.destinations), 2)
        self.assertTrue(all(item[0] == "upstream.invalid" for item in self.proxy.destinations))
        self.assertTrue(all(item[0] == "POST" for item in self.proxy.requests))

    def test_image_download_page_tokens_and_upload_use_same_proxy(self):
        self.assertEqual(multimodal.fetch_image_bytes("http://upstream.invalid/image"), b"image bytes")
        # Translate fixed HTTPS URLs to the local HTTP endpoint; retain the actual
        # SOCKS transport, headers, bodies and both resumable upload requests.
        def local_request(request, config, timeout, context):
            paths = {
                "https://gemini.google.com/app": "/page",
                "https://content-push.googleapis.com/upload/": "/start",
                "https://upload.invalid/finalize": "/finalize",
            }
            local = urllib.request.Request(
                "http://upstream.invalid" + paths[request.full_url],
                data=request.data, headers=dict(request.header_items()), method=request.get_method(),
            )
            return network.open_request(local, config, timeout, context)
        with mock.patch.object(multimodal, "open_request", side_effect=local_request):
            self.assertEqual(multimodal._get_page_tokens(), {"push_id": "push", "pctx": "pctx"})
            with mock.patch.object(multimodal, "_cached_page_tokens", return_value={}):
                self.assertEqual(multimodal.upload_image(b"png bytes"), "/uploaded/test-image")
        self.assertEqual(len(self.proxy.destinations), 4)
        self.assertEqual(self.proxy.requests[-1][3], b"png bytes")

    def test_legacy_generation_and_image_helpers_use_warp(self):
        with mock.patch.object(self.legacy, "open_request", side_effect=lambda req, config, **kwargs:
            network.open_request(urllib.request.Request(
                "http://upstream.invalid/generate", data=req.data,
                headers=dict(req.header_items()), method=req.get_method(),
            ), config, **kwargs)
        ):
            raw = self.legacy.gemini_stream_generate("hello", 1, 4)
            self.assertEqual(self.legacy.extract_response_text(raw), "hello")
        CONFIG["warp_enabled"] = False
        with mock.patch.object(multimodal, "upload_image", return_value="/image"):
            self.assertEqual(self.legacy.upload_images([(b"png", "image/png")]), ["/image"])
        self.assertTrue(CONFIG["warp_enabled"])

    def test_proxy_failure_never_falls_back_to_urlopen(self):
        with mock.patch.object(network, "create_httpx_client", side_effect=RuntimeError("proxy unavailable")), \
                mock.patch("urllib.request.urlopen") as direct:
            with self.assertRaisesRegex(RuntimeError, "proxy unavailable"):
                self.request("/image")
            direct.assert_not_called()

    def test_http_proxy_cannot_be_bypassed_by_no_proxy(self):
        CONFIG.update(warp_proxy="http://proxy.invalid:8080")
        opener = mock.Mock()
        with mock.patch("urllib.request.build_opener", return_value=opener) as build:
            self.request("/image")
        handler = build.call_args.args[0]
        request = urllib.request.Request("https://upstream.invalid/image")
        handler.proxy_open(request, CONFIG["warp_proxy"], "https")
        self.assertEqual(request.host, "proxy.invalid:8080")
        self.assertEqual(request._tunnel_host, "upstream.invalid")

    def test_cli_overrides_environment_and_config(self):
        parser = argparse.ArgumentParser()
        network.add_proxy_arguments(parser)
        network.configure_proxy(CONFIG, parser.parse_args(["--proxy", "http://localhost:7890"]))
        self.assertFalse(CONFIG["warp_enabled"])
        self.assertEqual(network.get_proxy(CONFIG), "http://localhost:7890")
        network.configure_proxy(CONFIG, parser.parse_args(["--warp", "--warp-proxy", "http://localhost:1080"]))
        self.assertEqual(network.get_proxy(CONFIG), "http://localhost:1080")

    def test_check_warp_validates_exit_and_reports_country(self):
        for status in ("on", "plus", "off"):
            with self.subTest(status=status):
                response = io.BytesIO(("warp={}\nip=203.0.113.1\nloc=SG\n".format(status)).encode())
                with mock.patch.object(network, "open_request", return_value=response):
                    if status == "off":
                        with self.assertRaisesRegex(RuntimeError, "not connected"):
                            network.check_warp(CONFIG)
                    else:
                        self.assertIn("IP: 203.0.113.1 | Country: SG", network.check_warp(CONFIG))


if __name__ == "__main__":
    unittest.main()
