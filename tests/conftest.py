"""Fixtures for the tests that were written to be run as scripts.

Several test files predate being run under pytest: they take their input
as a function argument and a `main()` at the bottom supplies it. pytest
reads that argument as a request for a fixture, finds none, and reports
an error that looks like a broken test but is only a missing argument.

Supplying the arguments here is what turns them back into tests, and it
keeps the files runnable as scripts, which is how they are usually used.
"""

import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
# The repo root as well, so `from tests import ...` below resolves. Running
# as `python -m pytest` put the working directory on the path and hid the
# need for this; `uv run pytest` does not, and the blame/bisect fixtures
# failed to import under it.
sys.path.insert(0, str(_ROOT))


@pytest.fixture(params=[False, True], ids=["single-file", "sharded"])
def shard(request):
    """Both safetensors layouts: one file, and an index over several."""
    return request.param


@pytest.fixture(scope="module")
def tmp(request):
    """A scratch directory, named after the module that asked for it.

    Module-scoped rather than per-test because the scripts these tests
    came from build one model into a shared directory and then run every
    check against it, which is what their `main()` does.
    """
    path = Path(__file__).parent / f"tmp_{request.module.__name__.split('.')[-1]}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def log_path():
    """A training log with real snapshots, for blame and bisect to walk.

    Module-scoped because producing it means actually training a small
    model for sixteen steps, and every test in that file wants the same
    one -- which is what the script's `main()` does too.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from tests import test_blame_bisect as suite

    shutil.rmtree(suite.TMP, ignore_errors=True)
    suite.TMP.mkdir(parents=True, exist_ok=True)
    try:
        yield suite.train_and_log()
    finally:
        shutil.rmtree(suite.TMP, ignore_errors=True)
