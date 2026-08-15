"""Verify the packed weight index changes when loading happens, not what.

The index exists to move work from every run to one run. So the property
that matters is that it is *invisible*: a model read through it must place
every weight exactly where reading it the slow way would have, and produce
the same logits doing so. An index that is merely close is worse than none,
because it is fast and wrong and nothing says which.

The checks are chosen so each fails for its own reason:

  * logits from an indexed model against the same model without one, which
    is the whole claim in one line
  * the index is actually used, counted rather than assumed -- a fast path
    that silently is not taken would pass the check above and buy nothing
  * a float run ignores it, since the index holds one width and a run that
    asked for no packing asked for the original tensors
  * building, dropping and rebuilding leaves the model where it started
"""

import shutil
import sys
from pathlib import Path

import numpy as np

from reminis import packed_index
from reminis.backend import available_backends
from reminis.backend import select as select_backend
from reminis.infer import KVCache, Model

MODELS = Path(__file__).resolve().parents[1] / "models"
SOURCE = MODELS / "SmolLM-135M.f16.db"
TOKENS = [464, 3139, 286, 4881, 318]
BITS = 8

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  --  {detail}" if detail else ""))
        FAILURES.append(label)


def logits_of(db, pack_bits):
    """Final-position logits, and how many tensors came from the index."""
    model = Model(str(db), backend=select_backend("inference"),
                  pack_bits=pack_bits)
    cache = KVCache(model.cfg.n_layers, capacity=len(TOKENS) + 2,
                    backend=model.backend)
    out = np.asarray(model.forward(TOKENS, cache, offset=0))
    from_index = model.store.from_index
    model.close()
    return out, from_index


def main():
    if not SOURCE.exists():
        print(f"Missing {SOURCE.name}. Run: "
              f"reminis convert models/SmolLM-135M.f16.gguf")
        sys.exit(0)
    if not select_backend("inference").can_pack():
        print(f"skip  no backend here can multiply packed weights "
              f"({', '.join(available_backends())})")
        sys.exit(0)

    print("=" * 70)
    print("PACKED WEIGHT INDEX")
    print("=" * 70)

    tmp = Path(__file__).parent / "tmp_packed_index"
    tmp.mkdir(exist_ok=True)
    db = tmp / "indexed.db"
    try:
        shutil.copy(SOURCE, db)
        before_bytes = db.stat().st_size

        print("\nWithout an index")
        plain, plain_hits = logits_of(db, BITS)
        check("nothing comes from an index that does not exist",
              plain_hits == 0, f"{plain_hits} tensors did")

        print("\nBuilding it")
        summary = packed_index.build(str(db), bits=BITS)
        check("the build reports the tensors it wrote",
              summary["tensors"] > 0, f"wrote {summary['tensors']}")
        check("and the width it wrote them at",
              summary["bits"] == BITS, f"said {summary['bits']}")

        import sqlite3
        conn = sqlite3.connect(db)
        layouts = packed_index.read_layouts(conn)
        conn.close()
        check("every written tensor has a layout to read it back",
              len(layouts) == summary["tensors"],
              f"{len(layouts)} layouts for {summary['tensors']} tensors")

        print("\nWith it")
        indexed, indexed_hits = logits_of(db, BITS)
        check("the index is what the weights came from",
              indexed_hits == len(layouts),
              f"{indexed_hits} of {len(layouts)} tensors")

        # The index stores exactly what the slow path would have produced,
        # so this is not "close enough" -- it is the same arithmetic on the
        # same bytes, and anything but equality means a layout is wrong.
        worst = float(np.max(np.abs(plain - indexed)))
        check("the logits are the ones the slow path gives", worst == 0.0,
              f"largest difference {worst:.3e}")
        check("and so is the token they choose",
              int(np.argmax(plain)) == int(np.argmax(indexed)))

        print("\nA run that wants floats")
        floats, float_hits = logits_of(db, None)
        check("ignores the index entirely", float_hits == 0,
              f"{float_hits} tensors came from it")
        check("and is not the packed model", float(np.max(np.abs(floats - plain))) > 0)

        print("\nDropping it")
        check("drop says there was one", packed_index.drop(str(db)))
        conn = sqlite3.connect(db)
        check("and leaves none behind",
              packed_index.read_layouts(conn) == {})
        conn.close()
        after, after_hits = logits_of(db, BITS)
        check("the model is where it started", after_hits == 0
              and float(np.max(np.abs(after - plain))) == 0.0)
        check("and the space came back",
              db.stat().st_size <= before_bytes * 1.02,
              f"{db.stat().st_size / 1e6:.1f} MB against "
              f"{before_bytes / 1e6:.1f} MB")
        check("dropping again is not an error",
              packed_index.drop(str(db)) is False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PACKED INDEX TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
