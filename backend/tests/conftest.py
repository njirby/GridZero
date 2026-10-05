import os, sys
# Make the repo root importable (backend.*) and keep tests off opencode.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPENCODE_DISABLE", "1")

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
