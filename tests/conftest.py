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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(params=[False, True], ids=["single-file", "sharded"])
def shard(request):
    """Both safetensors layouts: one file, and an index over several."""
    return request.param


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
