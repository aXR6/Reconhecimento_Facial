import os
import sys
import types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.google_search as gs


def test_run_google_search(monkeypatch, tmp_path):
    called = {}

    def dummy_post(url, files, allow_redirects):
        called["posted"] = True

        class Resp:
            status_code = 302
            headers = {"Location": "http://example.com"}

        return Resp()

    monkeypatch.setattr(gs.requests, "post", dummy_post)
    monkeypatch.setattr(gs.webbrowser, "open", lambda url: called.update({"open": url}))

    img = tmp_path / "img.jpg"
    img.write_text("x")

    urls = gs.run_google_search([str(img)])

    assert called.get("posted")
    assert called.get("open") == "http://example.com"
    assert urls == ["http://example.com"]
