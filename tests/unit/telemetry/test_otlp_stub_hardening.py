"""The test OTLP receiver itself must not be a weak server (CodeQL #10, #11 on the mirror)."""

from __future__ import annotations

import http.client
import ssl

from tests.support.otlp_stub import OtlpStub


def test_tls_stub_never_serves_below_tls_1_2():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1
    stub = OtlpStub(tls=context)
    try:
        stub.start()
        assert stub.tls.minimum_version >= ssl.TLSVersion.TLSv1_2
    finally:
        stub.stop()


def test_redirect_refuses_a_location_carrying_crlf():
    stub = OtlpStub(mode="redirect").start()
    try:
        stub.redirect_to = "http://127.0.0.1:1\r\nSet-Cookie: injected=1"
        conn = http.client.HTTPConnection("127.0.0.1", stub.port, timeout=5)
        conn.request("POST", "/v1/traces", body=b"", headers={"Content-Length": "0"})
        resp = conn.getresponse()
        assert resp.status == 400
        assert resp.getheader("Location") is None
        assert resp.getheader("Set-Cookie") is None
        conn.close()

        stub.redirect_to = "http://127.0.0.1:1"
        conn = http.client.HTTPConnection("127.0.0.1", stub.port, timeout=5)
        conn.request("POST", "/v1/traces", body=b"", headers={"Content-Length": "0"})
        resp = conn.getresponse()
        assert resp.status == 307
        assert resp.getheader("Location") == "http://127.0.0.1:1/v1/traces"
        conn.close()
    finally:
        stub.stop()
