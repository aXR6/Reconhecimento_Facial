import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import reconhecimento_facial.db as db


def _dummy_conn(fetch_rows=None, rowcount=1):
    class Cur:
        def execute(self, *a, **k):
            pass
        def fetchall(self):
            return fetch_rows or []
        @property
        def rowcount(self):
            return rowcount
    class Conn:
        def cursor(self):
            return Cur()
        def commit(self):
            pass

    class Wrapper:
        def __enter__(self):
            return Conn()

        def __exit__(self, exc_type, exc, tb):
            pass

    return Wrapper()


def test_list_people(monkeypatch):
    monkeypatch.setattr(db, "get_conn", lambda: _dummy_conn([("Alice",)]))
    assert db.list_people() == ["Alice"]


def test_delete_person(monkeypatch):
    monkeypatch.setattr(db, "get_conn", lambda: _dummy_conn())
    assert db.delete_person("Bob")
