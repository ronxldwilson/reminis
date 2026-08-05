"""Verify that `reminis merge` combines models the way each method says it does.

Two halves. The first builds tiny synthetic databases whose merged answer can
be written down by hand, so the arithmetic is checked against an expectation
rather than against itself. The second merges two real 135M checkpoints that
share an architecture, which is where the SQL alignment, the dtype round-trip
through F16, and the GGUF export actually get exercised.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

from reminis.converter import SCHEMA
from reminis.merge import _ties, merge_models

MODELS_DIR = Path(__file__).parent.parent / "models"
TMP_DIR = Path(__file__).parent / "tmp_merge"

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  --  {detail}" if detail else ""))
        FAILURES.append(label)


def close(label, got, want, tol=1e-5):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    ok = got.shape == want.shape and np.allclose(got, want, atol=tol)
    check(label, ok, f"got {got.tolist()}, want {want.tolist()}")


def build_db(path, tensors, meta=None):
    """Write a minimal model database. `tensors` maps name -> (dtype, array)."""
    Path(path).unlink(missing_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO model_meta (key, value, type) VALUES (?, ?, ?)",
        [("general.architecture", "test", "string"), ("general.name", "synthetic", "string")]
        + [(k, v, "string") for k, v in (meta or {}).items()],
    )
    for name, (dtype, arr) in tensors.items():
        if dtype == "F32":
            blob = np.asarray(arr, dtype=np.float32).tobytes()
        elif dtype == "F16":
            blob = np.asarray(arr, dtype=np.float16).tobytes()
        else:
            # A stand-in for a quantized payload: raw bytes with no float meaning.
            blob = bytes(arr)
        n = len(arr)
        conn.execute(
            "INSERT INTO tensors (name, shape, dtype, dtype_id, n_elements, n_bytes, data) "
            "VALUES (?,?,?,?,?,?,?)",
            (name, f"[{n}]", dtype, 0, n, len(blob), blob),
        )
    conn.commit()
    conn.close()
    return str(path)


def read_tensor(db, name):
    conn = sqlite3.connect(db)
    dtype, blob = conn.execute(
        "SELECT dtype, data FROM tensors WHERE name = ?", (name,)
    ).fetchone()
    conn.close()
    np_dtype = {"F32": np.float32, "F16": np.float16}[dtype]
    return np.frombuffer(blob, dtype=np_dtype).astype(np.float64)


def read_meta(db, key):
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT value FROM model_meta WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def expect_error(label, fragment, fn):
    try:
        fn()
    except (ValueError, FileNotFoundError) as exc:
        check(label, fragment.lower() in str(exc).lower(), f"message was: {exc}")
        return
    check(label, False, "no error raised")


def test_methods():
    print("\nSynthetic merges (the answer is computed by hand)")

    a = build_db(TMP_DIR / "a.db", {"w": ("F32", [1.0, 2.0, 3.0, 4.0])})
    b = build_db(TMP_DIR / "b.db", {"w": ("F32", [5.0, 6.0, 7.0, 8.0])})
    base = build_db(TMP_DIR / "base.db", {"w": ("F32", [1.0, 1.0, 1.0, 1.0])})
    out = str(TMP_DIR / "out.db")

    merge_models([a, b], out, method="linear", verbose=False)
    close("linear, equal weights, is the average", read_tensor(out, "w"), [3, 4, 5, 6])

    merge_models([a, b], out, method="linear", weights=[3, 1], verbose=False)
    close("linear weights normalise to sum 1", read_tensor(out, "w"), [2, 3, 4, 5])

    merge_models([a, b], out, method="slerp", t=0.0, verbose=False)
    close("slerp at t=0 is the first model", read_tensor(out, "w"), [1, 2, 3, 4])
    merge_models([a, b], out, method="slerp", t=1.0, verbose=False)
    close("slerp at t=1 is the second model", read_tensor(out, "w"), [5, 6, 7, 8])

    # Slerp travels along the arc, so the midpoint is off the straight line
    # between the two and its norm is not the average of the two norms.
    merge_models([a, b], out, method="slerp", t=0.5, verbose=False)
    mid = read_tensor(out, "w")
    lerp = np.array([3, 4, 5, 6], dtype=np.float64)
    check("slerp midpoint is not the linear midpoint", not np.allclose(mid, lerp, atol=1e-3),
          f"got {mid.tolist()}")
    check("slerp midpoint stays between the two norms",
          np.linalg.norm(a_norm := np.array([1, 2, 3, 4.0])) <= np.linalg.norm(mid)
          <= np.linalg.norm(np.array([5, 6, 7, 8.0])),
          f"|mid|={np.linalg.norm(mid):.4f}")

    # Two identical models: the arc has no length, so slerp must fall back to
    # a linear blend rather than dividing by a vanishing sine.
    a2 = build_db(TMP_DIR / "a2.db", {"w": ("F32", [1.0, 2.0, 3.0, 4.0])})
    merge_models([a, a2], out, method="slerp", t=0.5, verbose=False)
    close("slerp between identical models returns them unchanged",
          read_tensor(out, "w"), [1, 2, 3, 4])

    # task arithmetic: base + (a-base) + (b-base) = [1,1,1,1] + [0,1,2,3] + [4,5,6,7]
    merge_models([a, b], out, method="task-arithmetic", base=base,
                 weights=[1, 1], verbose=False)
    close("task arithmetic adds both task vectors to the base",
          read_tensor(out, "w"), [5, 7, 9, 11])

    merge_models([a, b], out, method="task-arithmetic", base=base,
                 weights=[1, 1], scale=0.5, verbose=False)
    close("--scale halves the combined task vector",
          read_tensor(out, "w"), [3, 4, 5, 6])

    # A single fine-tune against its base: scale 1 reproduces it, scale -1
    # subtracts the task vector instead of adding it.
    merge_models([a], out, method="task-arithmetic", base=base, verbose=False)
    close("one model at scale 1 reproduces that model",
          read_tensor(out, "w"), [1, 2, 3, 4])
    merge_models([a], out, method="task-arithmetic", base=base,
                 scale=-1.0, verbose=False)
    close("a negative scale subtracts the fine-tune from the base",
          read_tensor(out, "w"), [1, 0, -1, -2])

    # ties end to end, no trimming, so only the sign election is in play.
    c = build_db(TMP_DIR / "c.db", {"w": ("F32", [2.0, 0.0, 3.0, 1.0])})
    d = build_db(TMP_DIR / "d.db", {"w": ("F32", [4.0, 3.0, 0.0, 1.0])})
    zero = build_db(TMP_DIR / "zero.db", {"w": ("F32", [1.0, 1.0, 1.0, 1.0])})
    merge_models([c, d], out, method="ties", base=zero, density=1.0, verbose=False)
    # task vectors [1,-1,2,0] and [3,2,-1,0]; elected sign [+,+,+,0];
    # agreeing means 4/2, 2/1, 2/1, none -> [2,2,2,0]; plus the base.
    close("ties averages only the entries that agree with the elected sign",
          read_tensor(out, "w"), [3, 3, 3, 1])


def test_ties_trim():
    print("\nTIES trimming")
    tv = np.array([4.0, -1.0, 2.0, 0.5], dtype=np.float32)
    # density 0.5 of 4 entries keeps the two largest by magnitude: 4 and 2.
    kept = _ties([tv], [1.0], density=0.5)
    close("trimming keeps the largest entries and zeroes the rest",
          kept, [4, 0, 2, 0])

    check("density 1.0 keeps everything",
          np.allclose(_ties([tv], [1.0], density=1.0), tv, atol=1e-6))

    # Entries where the models point opposite ways and cancel exactly are
    # dropped, not averaged toward zero by the disagreeing side.
    up = np.array([1.0, 5.0], dtype=np.float32)
    down = np.array([-1.0, 1.0], dtype=np.float32)
    merged = _ties([up, down], [1.0, 1.0], density=1.0)
    check("an exact tie between opposite signs contributes nothing",
          merged[0] == 0.0, f"got {merged[0]}")
    close("the agreeing entry is the weighted mean of the agreers",
          merged[1:], [3.0])


def test_rejections():
    print("\nRefusals (a bad merge must fail before it writes a file)")

    a = str(TMP_DIR / "a.db")
    b = str(TMP_DIR / "b.db")
    out = str(TMP_DIR / "rejected.db")

    quant_a = build_db(TMP_DIR / "qa.db", {"w": ("Q4_K", bytes(range(32)))})
    quant_b = build_db(TMP_DIR / "qb.db", {"w": ("Q4_K", bytes(range(32, 64)))})
    expect_error("quantized tensors are refused, not averaged", "quantized",
                 lambda: merge_models([quant_a, quant_b], out, verbose=False))
    check("nothing was written when the merge was refused", not Path(out).exists())

    # The same quantized tensor in both models is a constant, not a conflict.
    quant_same = build_db(TMP_DIR / "qc.db", {"w": ("Q4_K", bytes(range(32))),
                                              "f": ("F32", [1.0, 2.0])})
    quant_same2 = build_db(TMP_DIR / "qd.db", {"w": ("Q4_K", bytes(range(32))),
                                               "f": ("F32", [3.0, 4.0])})
    summary = merge_models([quant_same, quant_same2], out, verbose=False)
    check("an identical quantized tensor is carried through untouched",
          summary["unchanged"] == ["w"] and summary["tensors_merged"] == 1,
          f"unchanged={summary['unchanged']} merged={summary['tensors_merged']}")
    close("and the float tensor beside it still merged",
          read_tensor(out, "f"), [2, 3])

    wide = build_db(TMP_DIR / "wide.db", {"w": ("F32", [1.0] * 8)})
    expect_error("mismatched shapes are refused", "different shapes",
                 lambda: merge_models([a, wide], out, verbose=False))

    expect_error("slerp rejects three models", "exactly two",
                 lambda: merge_models([a, b, a], out, method="slerp", verbose=False))
    expect_error("ties without a base is refused", "--base",
                 lambda: merge_models([a, b], out, method="ties", verbose=False))
    expect_error("one model is not a merge", "at least two",
                 lambda: merge_models([a], out, verbose=False))
    expect_error("a wrong weight count is refused", "must match",
                 lambda: merge_models([a, b], out, weights=[1.0], verbose=False))
    expect_error("merging in place is refused", "also an input",
                 lambda: merge_models([a, b], a, verbose=False))
    expect_error("an unknown method is refused", "unknown merge method",
                 lambda: merge_models([a, b], out, method="soup", verbose=False))
    expect_error("a missing model is reported by path", "not found",
                 lambda: merge_models([a, str(TMP_DIR / "nope.db")], out, verbose=False))


def test_structure_mismatch():
    print("\nTensors that only one model has")
    a = build_db(TMP_DIR / "sa.db", {"w": ("F32", [1.0, 1.0]), "only_a": ("F32", [7.0])})
    b = build_db(TMP_DIR / "sb.db", {"w": ("F32", [3.0, 3.0]), "only_b": ("F32", [9.0])})
    out = str(TMP_DIR / "sout.db")

    summary = merge_models([a, b], out, verbose=False)
    close("the shared tensor merged", read_tensor(out, "w"), [2, 2])
    check("a tensor only the first model has is carried through",
          summary["passthrough"] == ["only_a"], str(summary["passthrough"]))
    close("...with its value intact", read_tensor(out, "only_a"), [7])
    check("a tensor only a later model has is reported as dropped",
          summary["dropped"] == ["only_b"], str(summary["dropped"]))

    conn = sqlite3.connect(out)
    names = {r[0] for r in conn.execute("SELECT name FROM tensors")}
    conn.close()
    check("...and is genuinely absent from the result", "only_b" not in names)


def test_mixed_dtypes():
    print("\nModels stored in different dtypes")
    a = build_db(TMP_DIR / "ma.db", {"w": ("F32", [1.0, 2.0])})
    b = build_db(TMP_DIR / "mb.db", {"w": ("F16", [3.0, 4.0])})
    out = str(TMP_DIR / "mout.db")

    summary = merge_models([a, b], out, verbose=False)
    close("an F32 and an F16 model average correctly", read_tensor(out, "w"), [2, 3])
    check("the mixed dtype is reported", summary["mixed_dtype"] == 1)

    conn = sqlite3.connect(out)
    dtype = conn.execute("SELECT dtype FROM tensors WHERE name='w'").fetchone()[0]
    conn.close()
    check("the result keeps the first model's dtype", dtype == "F32", dtype)


def test_provenance():
    print("\nProvenance")
    a = str(TMP_DIR / "a.db")
    b = str(TMP_DIR / "b.db")
    out = str(TMP_DIR / "prov.db")
    merge_models([a, b], out, method="linear", weights=[3, 1], verbose=False)

    check("the method is recorded", read_meta(out, "reminis.merge.method") == "linear")
    check("the sources are recorded",
          read_meta(out, "reminis.merge.sources") == '["a.db", "b.db"]',
          str(read_meta(out, "reminis.merge.sources")))
    check("the normalised weights are recorded",
          read_meta(out, "reminis.merge.weights") == "[0.75, 0.25]",
          str(read_meta(out, "reminis.merge.weights")))
    check("the model no longer claims to be its first parent",
          read_meta(out, "general.name") == "synthetic (linear merge)",
          str(read_meta(out, "general.name")))


def test_chunking_matches_whole_tensor():
    """Blocked processing must give the answer the whole-tensor code gives.

    Tensors are combined in blocks so that peak memory does not depend on
    how big the largest tensor is. Two of the four methods need a quantity
    measured over the entire tensor before any block can be handled -- an
    angle for slerp, a trim threshold for ties -- and those are the ones a
    chunking bug would silently corrupt. So the block size is shrunk to a
    few dozen elements here, forcing many blocks over a tensor small enough
    that the reference answer can be computed outright.
    """
    print("\nChunked merges against whole-tensor merges")
    from reminis import merge as M

    rng = np.random.default_rng(11)
    n = 5000
    base_w = rng.standard_normal(n).astype(np.float32)
    a_w = (base_w + rng.standard_normal(n) * 0.1).astype(np.float32)
    b_w = (base_w + rng.standard_normal(n) * 0.1).astype(np.float32)

    a = build_db(TMP_DIR / "ca.db", {"w": ("F32", a_w)})
    b = build_db(TMP_DIR / "cb.db", {"w": ("F32", b_w)})
    base = build_db(TMP_DIR / "cbase.db", {"w": ("F32", base_w)})
    out = str(TMP_DIR / "cout.db")

    original = M.CHUNK_ELEMENTS
    try:
        for chunk in (n * 2, 64, 999):
            M.CHUNK_ELEMENTS = chunk
            label = "one block" if chunk > n else f"{-(-n // chunk)} blocks"

            M.merge_models([a, b], out, method="linear", verbose=False)
            close(f"linear, {label}", read_tensor(out, "w"), (a_w + b_w) / 2, tol=1e-6)

            M.merge_models([a, b], out, method="slerp", t=0.3, verbose=False)
            # The reference is the whole-tensor slerp, with no precomputed
            # statistics -- the path the chunked version has to reproduce.
            want = M._slerp(a_w.copy(), b_w.copy(), 0.3)
            close(f"slerp, {label}", read_tensor(out, "w"), want, tol=1e-5)

            M.merge_models([a, b], out, method="task-arithmetic", base=base,
                           weights=[0.7, 0.4], scale=0.9, verbose=False)
            want = base_w + 0.9 * (0.7 * (a_w - base_w) + 0.4 * (b_w - base_w))
            close(f"task arithmetic, {label}", read_tensor(out, "w"), want, tol=1e-5)

            for density in (0.2, 0.5, 0.83):
                M.merge_models([a, b], out, method="ties", base=base,
                               density=density, verbose=False)
                want = base_w + M._ties([a_w - base_w, b_w - base_w],
                                        [1.0, 1.0], density)
                close(f"ties d={density}, {label}", read_tensor(out, "w"), want, tol=1e-5)
    finally:
        M.CHUNK_ELEMENTS = original


def test_trim_cutoff():
    """The blocked trim threshold must equal what np.partition returns.

    Trimming keeps the largest fraction of a task vector by magnitude, which
    is a rank over every entry. Found in passes it is a histogram narrowed
    until the band around the threshold is small enough to rank exactly, and
    "exactly" is the claim being checked: an approximate threshold would
    keep the wrong entries and no test of the merged values would say why.
    """
    print("\nTrim threshold found in passes")
    from reminis import merge as M

    rng = np.random.default_rng(5)
    cases = [
        ("gaussian", rng.standard_normal(20000).astype(np.float32)),
        ("heavy-tailed", (rng.standard_normal(20000) ** 5).astype(np.float32)),
        # Many entries sharing one magnitude is the case that would overflow
        # a single collection pass, so the band has to keep narrowing.
        ("mostly identical", np.r_[np.full(19000, 0.5), rng.standard_normal(1000)
                                   ].astype(np.float32)),
        ("all zeros", np.zeros(5000, dtype=np.float32)),
    ]

    zero = np.zeros_like(cases[0][1])
    original = M.CHUNK_ELEMENTS
    try:
        M.CHUNK_ELEMENTS = 512
        for label, vec in cases:
            base = build_db(TMP_DIR / "tb.db", {"w": ("F32", np.zeros(vec.size, np.float32))})
            model = build_db(TMP_DIR / "tm.db", {"w": ("F32", vec)})
            conn = sqlite3.connect(str(model))
            conn.execute("ATTACH DATABASE ? AS m0", (str(Path(model).resolve()),))
            conn.execute("ATTACH DATABASE ? AS base", (str(Path(base).resolve()),))
            reader = M._TensorSlicer(conn, "m0", "w")
            base_reader = M._TensorSlicer(conn, "base", "w")
            for density in (0.1, 0.25, 0.5, 0.9):
                got = M._trim_cutoff(reader, base_reader, density, vec.size)
                k = max(1, int(round(vec.size * density)))
                want = float(np.partition(np.abs(vec), vec.size - k)[vec.size - k])
                check(f"{label}, density {density}", got == want,
                      f"got {got!r}, np.partition says {want!r}")
            reader.close()
            base_reader.close()
            conn.close()
    finally:
        M.CHUNK_ELEMENTS = original


def test_real_models():
    """Merge two real checkpoints that share an architecture."""
    base_db = MODELS_DIR / "SmolLM-135M.f16.db"
    instruct_db = MODELS_DIR / "SmolLM-135M-Instruct.f16.db"
    if not (base_db.exists() and instruct_db.exists()):
        print("\nSkipping the real-model merge (SmolLM-135M f16 databases not present)")
        return

    print(f"\nReal merge: {base_db.name} + {instruct_db.name}")
    out = str(TMP_DIR / "smollm-soup.db")
    summary = merge_models([str(base_db), str(instruct_db)], out,
                           method="linear", verbose=False)

    print(f"  {summary['tensors_merged']} tensors, "
          f"{summary['parameters_merged']:,} parameters, "
          f"mean drift {summary['mean_drift'] * 100:.2f}%")

    check("every tensor merged", summary["tensors_merged"] > 0)
    check("nothing was dropped or left behind",
          not summary["dropped"] and not summary["passthrough"],
          f"dropped={len(summary['dropped'])} passthrough={len(summary['passthrough'])}")
    check("the merge actually moved the weights",
          0 < summary["mean_drift"] < 1, f"{summary['mean_drift']}")

    # Spot-check one tensor against the average computed straight from the
    # two source databases -- in F16, since that is what gets written back.
    name = "blk.0.attn_q.weight"
    left = read_tensor(str(base_db), name)
    right = read_tensor(str(instruct_db), name)
    want = np.asarray((left + right) / 2, dtype=np.float16).astype(np.float64)
    close(f"{name} is the F16 average of the two", read_tensor(out, name), want, tol=1e-3)

    # A merged database is still a model: it must export back to GGUF.
    from reminis.converter import sqlite_to_gguf
    gguf_path = str(TMP_DIR / "smollm-soup.gguf")
    sqlite_to_gguf(out, gguf_path, verbose=False)
    exported = Path(gguf_path).stat().st_size
    original = base_db.stat().st_size
    check("the merged model exports back to a GGUF of the expected size",
          abs(exported - original) / original < 0.01,
          f"{exported} vs {original}")

    # Task arithmetic against the base, at scale 1, must rebuild the
    # fine-tune: base + (instruct - base) cancels. Every one of the 135M
    # weights is compared.
    #
    # The comparison is by value, not by bytes, and the difference matters.
    # Where the fine-tune stores -0.0, adding a zero task vector to a +0.0
    # base yields +0.0 -- a different bit pattern for the same number. That
    # is what floating-point addition does, so the test asserts what is
    # actually true rather than a byte equality merging cannot promise.
    rebuilt = str(TMP_DIR / "smollm-rebuilt.db")
    merge_models([str(instruct_db)], rebuilt, method="task-arithmetic",
                 base=str(base_db), verbose=False)
    conn_a, conn_b = sqlite3.connect(rebuilt), sqlite3.connect(str(instruct_db))
    mismatched, byte_diff = [], 0
    for tname, blob in conn_a.execute("SELECT name, data FROM tensors"):
        ref = conn_b.execute("SELECT data FROM tensors WHERE name = ?", (tname,)).fetchone()[0]
        if blob == ref:
            continue
        byte_diff += 1
        got = np.frombuffer(blob, dtype=np.float16)
        want = np.frombuffer(ref, dtype=np.float16)
        if not np.array_equal(got, want):
            mismatched.append(tname)
    conn_a.close()
    conn_b.close()
    check("base + (finetune - base) reproduces every weight of the fine-tune",
          not mismatched, f"{len(mismatched)} tensors differ, e.g. {mismatched[:3]}")
    print(f"  ({byte_diff} tensors differ only in the sign of zeros)")

    # And a slerp of the same pair should differ from the linear soup.
    slerp_out = str(TMP_DIR / "smollm-slerp.db")
    merge_models([str(base_db), str(instruct_db)], slerp_out,
                 method="slerp", t=0.5, verbose=False)
    check("slerp gives a different model than the linear average",
          not np.allclose(read_tensor(slerp_out, name), read_tensor(out, name), atol=1e-4))


def main():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MERGE TESTS")
    print("=" * 70)

    try:
        test_methods()
        test_ties_trim()
        test_rejections()
        test_structure_mismatch()
        test_mixed_dtypes()
        test_provenance()
        test_chunking_matches_whole_tensor()
        test_trim_cutoff()
        test_real_models()
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} MERGE TESTS FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 70)
        sys.exit(1)
    print("ALL MERGE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
