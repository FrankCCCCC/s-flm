"""Shared pytest fixtures / path setup for the test suite.

Existing tests (e.g. `test_geo_bridge_overflow.py`) do `from conftest import
REPO_ROOT`, which requires both this module to exist AND the repo root to be on
`sys.path` so that top-level modules (`geo_bridge`, `algo`, `samplers`,
`trainer_base`, `main`, `models`) import. We insert the repo root (the parent of
this `tests/` dir) at the front of `sys.path` at import time so collection works
regardless of the pytest invocation directory.
"""
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
  config.addinivalue_line(
    'markers', 'network: test hits the network (HF Hub download)')
