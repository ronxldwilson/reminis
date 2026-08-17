"""Verify the materialized expert index.

The index is a second physical copy of weights the database already holds,
so every test here is some form of the same question: does reading through
it give what reading around it gives? A wrong answer would not look wrong
-- a model whose experts are subtly corrupted still emits fluent text --
so nothing short of comparing against the path that does not use the index
would catch it.

The speed is not tested. It is measured, and measurements belong in the
README where they can carry their conditions with them.
"""

import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from reminis import expert_index
from reminis.backend import NumpyBackend, best_group, select

MODELS_DIR = Path(__file__).parent.parent / "models"
# A mixture of experts small enough to index in a few seconds.
MOE = MODELS_DIR / "granite-3.1-1b-a400m-instruct-Q4_K_M.db"
DENSE = MODELS_DIR / "SmolLM-135M.f16.db"

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  --  {detail}" if detail else ""))
        FAILURES.append(label)


def have_mlx():
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except Exception:
        return False


def test_group_sizes():
    """The group size has to divide the row, whatever was asked for."""
    print("\nChoosing a quantization group")
    check("a 4096-wide row takes the largest group", best_group(4096) == 128)
    # gpt-oss is 2880 wide, which 128 does not divide -- getting this wrong
    # was a hard failure at index build rather than a silent one.
    check("gpt-oss's 2880 falls back to 64", best_group(2880) == 64)
    check("896 takes 128", best_group(896) == 128)
    check("576 takes 64", best_group(576) == 64)
    check("a limit is respected", best_group(4096, limit=32) == 32)
    check("an awkward width still returns something usable",
          best_group(100) == 32)


def test_reserve_is_optional():
    """Pinning memory is an Apple-silicon notion; elsewhere it does nothing."""
    print("\nReserving memory")
    check("numpy's reserve is a no-op that returns a number",
          isinstance(NumpyBackend().reserve(1 << 30), int))
    if have_mlx():
        got = select(requested="mlx").reserve(1 << 30)
        check("mlx returns the previous limit", isinstance(got, int))
        # Asking for more than the device has is refused by the runtime, so
        # it has to be clamped rather than passed through.
        huge = select(requested="mlx").reserve(1 << 60)
        check("an impossible request is clamped rather than raising",
              isinstance(huge, int))
        select(requested="mlx").reserve(0)


def test_build_and_read():
    """A built index must give back exactly what was packed into it."""
    print("\nBuilding an index and reading it back")
    if not MOE.exists():
        print("  skip  no mixture-of-experts model to index")
        return
    if not have_mlx():
        print("  skip  no backend that can multiply packed weights")
        return
    import mlx.core as mx

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "moe.db")
        shutil.copy(MOE, path)
        summary = expert_index.build(path, bits=4, group_size=128)

        conn = sqlite3.connect(path)
        layouts = expert_index.read_layouts(conn)
        check("every expert tensor is indexed",
              len(layouts) == summary["tensors"], f"{len(layouts)} layouts")
        check("the expert count matches the layouts",
              sum(l.n_experts for l in layouts.values()) == summary["experts"])

        name = sorted(layouts)[0]
        layout = layouts[name]
        check("the group size divides the row",
              layout.cols % layout.group_size == 0,
              f"{layout.cols} % {layout.group_size}")

        # Ids are assigned so that a tensor's experts are consecutive, which
        # is what makes the read pattern sequential rather than scattered.
        check("experts of one tensor are contiguous",
              layout.row_id(1) - layout.row_id(0) == 1)
        others = [l for n, l in layouts.items() if n != name]
        check("tensors do not overlap",
              all(l.base_id >= layout.base_id + layout.n_experts
                  or l.base_id + l.n_experts <= layout.base_id
                  for l in others))

        # The real check: the stored block, split and handed to the backend,
        # dequantizes to the same numbers as re-packing the tensor here.
        backend = select(requested="mlx")
        blob = conn.execute("SELECT block FROM expert_index WHERE id = ?",
                            (layout.row_id(3),)).fetchone()[0]
        check("the block is the size the layout says",
              len(blob) == layout.block_bytes,
              f"{len(blob)} vs {layout.block_bytes}")

        words, scales, biases = layout.split(blob)
        check("the codes come back as unsigned words",
              words.dtype == np.uint32)
        check("the scales come back at half precision",
              scales.dtype == np.float16 and biases.dtype == np.float16)
        check("the scales have one entry per group",
              scales.shape == (layout.rows, layout.cols // layout.group_size),
              str(scales.shape))

        from reminis.dtypes import to_float32_any

        row = conn.execute(
            "SELECT id, dtype, n_bytes FROM tensors WHERE name = ?",
            (name,)).fetchone()
        stride = row[2] // layout.n_experts
        with conn.blobopen("tensors", "data", row[0], readonly=True) as h:
            raw = h[3 * stride:4 * stride]
        source = to_float32_any(raw, row[1]).reshape(layout.rows, layout.cols)
        expected = backend.pack(backend.from_numpy(source), layout.bits,
                                layout.group_size, compact=True)
        stored = backend.adopt_packed(words, scales, biases, layout.bits,
                                      layout.group_size,
                                      (layout.rows, layout.cols), compact=True)
        check("the stored codes are the ones that were packed",
              np.array_equal(np.array(stored.q), np.array(expected.q)))
        check("the stored scales are the ones that were packed",
              np.array_equal(np.array(stored.scales), np.array(expected.scales)))
        conn.close()


def test_drop_is_reversible():
    """Dropping the index must leave a working model behind."""
    print("\nDropping an index")
    if not MOE.exists() or not have_mlx():
        print("  skip  no mixture-of-experts model, or no packing backend")
        return

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "moe.db")
        shutil.copy(MOE, path)
        before = Path(path).stat().st_size
        expert_index.build(path, bits=4)
        grown = Path(path).stat().st_size
        check("building the index makes the file bigger", grown > before)

        check("dropping reports that there was one", expert_index.drop(path))
        after = Path(path).stat().st_size
        # Without the VACUUM the pages are freed but the file keeps them,
        # and an index the size of the model is not a rounding error.
        check("the space comes back", after < grown * 0.9,
              f"{before / 1e6:.0f} -> {grown / 1e6:.0f} -> {after / 1e6:.0f} MB")

        conn = sqlite3.connect(path)
        check("no layouts remain", expert_index.read_layouts(conn) == {})
        check("the tensors are untouched",
              conn.execute("SELECT count(*) FROM tensors").fetchone()[0] > 0)
        conn.close()
        check("dropping again says there was nothing",
              not expert_index.drop(path))


def test_refusals():
    """A model with no experts has nothing to index."""
    print("\nModels with nothing to index")
    if not DENSE.exists():
        print("  skip  no dense model to refuse")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "dense.db")
        shutil.copy(DENSE, path)
        try:
            expert_index.build(path)
            check("a dense model is refused", False, "it built one anyway")
        except ValueError as exc:
            check("a dense model is refused",
                  "mixture-of-experts" in str(exc))

        conn = sqlite3.connect(path)
        check("a database with no index reads as having none",
              expert_index.read_layouts(conn) == {})
        conn.close()


def test_inference_agrees():
    """Reading experts through the index must not change the answer."""
    print("\nA forward pass with and without the index")
    if not MOE.exists() or not have_mlx():
        print("  skip  no mixture-of-experts model, or no packing backend")
        return

    from reminis.infer import KVCache, Model

    prompt = "The capital of France is Paris, and the capital of Germany is"

    def logits(path, cache_size):
        model = Model(path, expert_cache=cache_size, expert_bits=4)
        try:
            ids = model.tokenizer.encode(prompt, add_special=False)
            out = model.forward(
                ids, KVCache(model.cfg.n_layers, capacity=64,
                             backend=model.backend), 0)
            return out, model.store._expert_misses, bool(
                model.store._expert_layouts)
        finally:
            model.close()

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "moe.db")
        shutil.copy(MOE, path)

        whole, _, _ = logits(path, 0)
        streamed, misses, indexed = logits(path, 64)
        check("without an index, experts are read one at a time",
              misses > 0 and not indexed)

        expert_index.build(path, bits=4)
        through, indexed_misses, indexed = logits(path, 64)
        check("with an index, the index is what gets read", indexed)
        check("the same experts are fetched either way",
              indexed_misses == misses, f"{indexed_misses} vs {misses}")

        # Both on-demand paths quantize to 4 bits at the same group size, so
        # they should agree far more closely than either agrees with the
        # unquantized whole-tensor path.
        check("the index picks the same next token as reading around it",
              int(np.argmax(through)) == int(np.argmax(streamed)))
        gap = float(np.abs(through - streamed).max())
        check("the index's logits match reading around it", gap < 1e-2,
              f"largest difference {gap:.2e}")
        check("both still agree with the whole-tensor path",
              int(np.argmax(through)) == int(np.argmax(whole)),
              f"corr {np.corrcoef(through, whole)[0, 1]:.5f}")


def test_preload():
    """Preloading must leave nothing to fetch."""
    print("\nPreloading the whole index")
    if not MOE.exists() or not have_mlx():
        print("  skip  no mixture-of-experts model, or no packing backend")
        return

    from reminis.infer import KVCache, Model, UnsupportedModel

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "moe.db")
        shutil.copy(MOE, path)

        model = Model(path)
        try:
            model.preload_experts()
            check("preloading without an index is refused", False)
        except UnsupportedModel as exc:
            check("preloading without an index is refused",
                  "prepare" in str(exc))
        finally:
            model.close()

        summary = expert_index.build(path, bits=4)
        model = Model(path)
        try:
            loaded = model.preload_experts()
            check("every expert is loaded",
                  loaded["experts"] == summary["experts"])
            check("the cache is sized to hold them all",
                  model.store.expert_cache_size >= loaded["experts"])

            ids = model.tokenizer.encode("The capital of France is",
                                         add_special=False)
            cache = KVCache(model.cfg.n_layers, capacity=64,
                            backend=model.backend)
            before = model.store._expert_misses
            model.forward(ids, cache, 0)
            # The whole point: after preloading, a forward pass reads no
            # expert off the disk at all.
            check("a forward pass fetches nothing more",
                  model.store._expert_misses == before,
                  f"{model.store._expert_misses - before} fetches")
            check("and it is all hits",
                  model.store._expert_hits > 0)
        finally:
            model.close()


def test_mmap_decision():
    """Mapping the file is right when it fits and wrong when it does not."""
    print("\nDeciding whether to map the file")
    from reminis.infer import WeightStore

    import os

    memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    with tempfile.TemporaryDirectory() as tmp:
        small = Path(tmp) / "small.db"
        small.write_bytes(b"\x00" * 4096)
        check("a small file is mapped whole",
              WeightStore._mmap_size(str(small)) == 4096)

        # A file a quarter the size of memory or larger stops being free to
        # map: its pages compete with the weights the model wants to keep.
        big = Path(tmp) / "big.db"
        with open(big, "wb") as fh:
            fh.truncate(memory)
        check("a file the size of memory is not mapped",
              WeightStore._mmap_size(str(big)) == 0)
        check("a missing file does not raise",
              WeightStore._mmap_size(str(Path(tmp) / "nope.db")) == 0)
