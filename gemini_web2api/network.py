"""Shared HTTP/SOCKS proxy routing for both server entry points."""
import io
import os
import urllib.error
import urllib.parse
import urllib.request
import urllib.response


class _ExplicitProxyHandler(urllib.request.ProxyHandler):
    """Honor an explicitly selected proxy even if NO_PROXY matches the target."""

    def proxy_open(self, request, proxy, protocol):
        import base64
        parsed = urllib.parse.urlsplit(proxy)
        hostport = parsed.netloc.rsplit("@", 1)[-1]
        if parsed.username is not None:
            credentials = "{}:{}".format(
                urllib.parse.unquote(parsed.username),
                urllib.parse.unquote(parsed.password or ""),
            )
            request.add_header("Proxy-Authorization", "Basic " + base64.b64encode(credentials.encode()).decode())
        request.set_proxy(hostport, parsed.scheme)
        if protocol != parsed.scheme and protocol != "https":
            return self.parent.open(request, timeout=request.timeout)


def get_proxy(config):
    if config.get("warp_enabled", False):
        return config.get("warp_proxy") or "socks5://127.0.0.1:40000"
    return config.get("proxy")


def _httpx():
    try:
        import httpx
        return httpx
    except ImportError as exc:
        raise RuntimeError('SOCKS/WARP requires: pip install "httpx[socks]>=0.28"') from exc


def create_httpx_client(config, timeout=None, verify=True):
    httpx = _httpx()
    proxy = get_proxy(config)
    try:
        return httpx.Client(
            proxy=proxy,
            timeout=timeout if timeout is not None else config["request_timeout_sec"],
            verify=verify,
            follow_redirects=True,
            # Explicit routing must not be overridden by HTTPS_PROXY / NO_PROXY.
            trust_env=not bool(proxy),
        )
    except ImportError as exc:
        raise RuntimeError('SOCKS/WARP requires: pip install "httpx[socks]>=0.28"') from exc


def open_request(request, config, timeout, context):
    """Return a urllib-compatible response, including when using SOCKS."""
    proxy = get_proxy(config)
    if proxy and proxy.lower().startswith(("socks5://", "socks5h://")):
        with create_httpx_client(config, timeout=timeout, verify=context) as client:
            response = client.request(
                request.get_method(), request.full_url,
                headers=dict(request.header_items()), content=request.data,
            )
        body = io.BytesIO(response.content)
        if response.status_code >= 400:
            raise urllib.error.HTTPError(
                str(response.url), response.status_code, response.reason_phrase,
                response.headers, body,
            )
        return urllib.response.addinfourl(
            body, response.headers, str(response.url), response.status_code,
        )
    if proxy:
        opener = urllib.request.build_opener(
            _ExplicitProxyHandler({"http": proxy, "https": proxy}),
            urllib.request.HTTPSHandler(context=context),
        )
        return opener.open(request, timeout=timeout)
    return urllib.request.urlopen(request, context=context, timeout=timeout)


def add_proxy_arguments(parser):
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--proxy", help="HTTP or SOCKS5 proxy URL")
    routing.add_argument("--warp", action="store_true", help="Route outbound requests through WARP")
    parser.add_argument("--warp-proxy", help="WARP HTTP/SOCKS5 proxy URL (default: socks5://127.0.0.1:40000)")
    parser.add_argument("--check-warp", action="store_true", help="Check WARP exit IP/country and exit without starting the server")


def configure_proxy(config, args):
    enabled = os.environ.get("GEMINI_WARP_ENABLED")
    if enabled is not None:
        if enabled.lower() not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            raise ValueError("GEMINI_WARP_ENABLED must be true or false")
        config["warp_enabled"] = enabled.lower() in ("1", "true", "yes", "on")
    if os.environ.get("GEMINI_WARP_PROXY"):
        config["warp_proxy"] = os.environ["GEMINI_WARP_PROXY"]
    if args.warp_proxy:
        config["warp_proxy"] = args.warp_proxy
    if args.warp:
        config["warp_enabled"] = True
    if args.proxy:
        config["proxy"] = args.proxy
        config["warp_enabled"] = False
    if not isinstance(config.get("warp_enabled", False), bool):
        raise ValueError("warp_enabled must be a JSON boolean")
    proxy = get_proxy(config)
    if proxy:
        from urllib.parse import urlsplit
        parsed = urlsplit(proxy)
        if parsed.scheme not in ("http", "https", "socks5", "socks5h") or not parsed.hostname:
            raise ValueError("Proxy URL must use http://, https://, socks5:// or socks5h://")
        if parsed.scheme.startswith("socks"):
            # Fail at startup instead of retrying with a missing dependency.
            with create_httpx_client(config):
                pass


def check_warp(config):
    """Verify the actual proxy exit; never retry using a direct connection."""
    import ssl
    if not get_proxy(config):
        raise ValueError("Specify --warp or --proxy before checking the WARP exit")
    request = urllib.request.Request("https://www.cloudflare.com/cdn-cgi/trace")
    with open_request(request, config, timeout=30, context=ssl.create_default_context()) as response:
        fields = dict(
            line.split("=", 1) for line in response.read().decode().splitlines() if "=" in line
        )
    if fields.get("warp") not in ("on", "plus"):
        raise RuntimeError("WARP is not connected (warp={})".format(fields.get("warp", "unknown")))
    return "WARP: {warp} | IP: {ip} | Country: {loc}".format(
        warp=fields["warp"], ip=fields.get("ip", "unknown"), loc=fields.get("loc", "unknown"),
    )
