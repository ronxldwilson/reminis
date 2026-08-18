"""Per-group widths: the grouping, the accounting, and what the store packs.

A mix is only worth anything if the width asked for is the width applied. The
failure that matters is silent: a role map that names four groups and packs
three of them, or a group whose tensors quietly fall through to a default,
would still produce a table of plausible numbers and a mix that means
something other than what it says.

So these check the two halves separately. The grouping is checked against real
tensor names, including the ones that look like they belong to a group and do
not -- `attn_norm.weight` sits between `attn_q` and `attn_output` and is never
packed at any width. The accounting is checked against a database with known
element counts, so a mix's predicted size can be worked out by hand. And the
store is checked on a real model: the widths it reports per tensor, and the
fact that a group asked for 4 bits ends up smaller resident than the same
group at 8.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reminis import bitmix  # noqa: E402
from reminis.sweep import _predicted_bytes  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"
SMOL = MODELS / "SmolLM-135M.f16.db"


def test_roles_of_real_tensor_names():
    assert bitmix.role_of("token_embd.weight") == "embed"
    assert bitmix.role_of("output.weight") == "output"
    assert bitmix.role_of("blk.7.attn_q.weight") == "attn"
    assert bitmix.role_of("blk.7.attn_output.weight") == "attn"
    assert bitmix.role_of("blk.0.attn_qkv.weight") == "attn"
    assert bitmix.role_of("blk.31.ffn_down.weight") == "ffn"
    assert bitmix.role_of("blk.2.ffn_gate_exps.weight") == "ffn"
    assert bitmix.role_of("blk.4.ssm_out.weight") == "ssm"


def test_unpackable_tensors_have_no_role():
    """The ones that would be given a role by prefix alone, and must not be.

    `attn_norm` and `ffn_norm` start with the same word as the matrices
    around them. They are never packed, so a role would offer a width that
    nothing acts on -- and `_predicted_bytes` would then count them at that
    width and report a size the model never reaches.
    """
    for name in ("blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
                 "blk.0.attn_q.bias", "output_norm.weight", "rope_freqs.weight"):
        assert bitmix.role_of(name) is None, name
        assert not bitmix.is_packable(name), name


def test_roles_in_is_ordered_and_only_what_is_present():
    tied = ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_up.weight",
            "blk.0.attn_norm.weight"]
    assert bitmix.roles_in(tied) == ["embed", "attn", "ffn"]
    # An untied model gains `output`, and it sorts last because that is where
    # the forward pass reaches it.
    assert bitmix.roles_in([*tied, "output.weight"]) == [
        "embed", "attn", "ffn", "output"]
    assert bitmix.roles_in(["blk.0.attn_norm.weight"]) == []


def test_describe_reads_in_role_order():
    assert bitmix.describe({"ffn": 4, "embed": 8, "attn": 6}) == \
        "embed=8 attn=6 ffn=4"
    assert bitmix.describe({"embed": None, "ffn": 4}) == "embed=f16 ffn=4"
    assert bitmix.describe({}) == "(empty)"


def test_packed_index_shares_the_store_s_list():
    """The two must agree, and now they agree by construction.

    A tensor stored in the packed index that the store would not have packed
    comes back in a shape the forward pass does not expect. The expert
    matrices are the one deliberate difference: they have their own table.
    """
    from reminis import packed_index

    assert set(packed_index.PACKABLE) <= set(bitmix.PACKABLE_MATRIX)
    assert packed_index.PACKABLE_EMBED == bitmix.PACKABLE_EMBED
    assert set(bitmix.PACKABLE_MATRIX) - set(packed_index.PACKABLE) == \
        set(bitmix.EXPERT_MATRIX)


def _tiny_db(path, tensors):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE tensors (id INTEGER PRIMARY KEY, name TEXT, "
        "shape TEXT, dtype TEXT, dtype_id INTEGER, n_elements INTEGER, "
        "n_bytes INTEGER, data BLOB)"
    )
    conn.executemany(
        "INSERT INTO tensors (name, shape, dtype, dtype_id, n_elements, "
        "n_bytes, data) VALUES (?, '[8]', 'F16', 1, ?, 0, X'00')",
        tensors,
    )
    conn.commit()
    conn.close()


def test_predicted_bytes_for_a_mix_counts_each_group_at_its_own_width():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "m.db")
        _tiny_db(path, [("token_embd.weight", 1_000_000),
                        ("blk.0.attn_q.weight", 1_000_000),
                        ("blk.0.ffn_up.weight", 1_000_000),
                        ("blk.0.attn_norm.weight", 1_000_000)])

        # 8 + 6 + 4 bits for the three packable tensors, plus the two extra
        # bits a group scale costs each, and the norm at full precision.
        mix = {"embed": 8, "attn": 6, "ffn": 4}
        expected = 1_000_000 * (10 + 8 + 6 + 16) // 8
        assert _predicted_bytes(path, mix) == expected

        # A uniform width, spelled as a mix, has to agree with itself -- but
        # only on the packable tensors; the norm stays at 16 either way.
        uniform = {"embed": 4, "attn": 4, "ffn": 4}
        assert _predicted_bytes(path, uniform) == 1_000_000 * (6 * 3 + 16) // 8

        # A role the map leaves out is left wide rather than defaulted.
        assert _predicted_bytes(path, {"embed": 4}) == \
            1_000_000 * (6 + 16 * 3) // 8


@pytest.mark.skipif(not SMOL.exists(), reason="SmolLM-135M database not present")
def test_store_reports_the_width_each_tensor_will_be_packed_at():
    from reminis.weights import WeightStore

    store = WeightStore(str(SMOL), pack_bits={"embed": 8, "attn": 6, "ffn": 4})
    try:
        assert store.bits_for("token_embd.weight") == 8
        assert store.bits_for("blk.0.attn_q.weight") == 6
        assert store.bits_for("blk.0.ffn_up.weight") == 4
        # No role, so no width -- and nothing silently picks one for it.
        assert store.bits_for("blk.0.attn_norm.weight") is None
    finally:
        store.close()


@pytest.mark.skipif(not SMOL.exists(), reason="SmolLM-135M database not present")
def test_a_single_width_still_answers_the_same_for_every_tensor():
    """The mix is an addition, so the old spelling has to behave exactly."""
    from reminis.weights import WeightStore

    store = WeightStore(str(SMOL), pack_bits=6)
    try:
        for name in ("token_embd.weight", "blk.0.attn_q.weight",
                     "blk.0.ffn_up.weight", "blk.0.attn_norm.weight"):
            assert store.bits_for(name) == 6, name
        assert store._should_pack("blk.0.attn_norm.weight") is False
    finally:
        store.close()

    store = WeightStore(str(SMOL), pack_bits=None)
    try:
        assert store.bits_for("blk.0.attn_q.weight") is None
    finally:
        store.close()


@pytest.mark.skipif(not SMOL.exists(), reason="SmolLM-135M database not present")
def test_a_narrower_group_actually_loads_smaller():
    """The width asked for is the width applied, measured in bytes.

    Reported widths could be right while the packing ignored them, so this
    loads one tensor from each group at two mixes that differ in exactly one
    group and checks that the one narrowed shrank and the others did not.
    """
    from reminis.backend import select as select_backend
    from reminis.weights import WeightStore

    backend = select_backend("inference")
    if not backend.can_pack():
        pytest.skip("this backend holds no packed weights")

    names = ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_up.weight"]

    def sizes(mix):
        store = WeightStore(str(SMOL), backend=backend, pack_bits=mix)
        try:
            out = {}
            for name in names:
                value = store.get(name)
                out[name] = sum(
                    int(p.nbytes) for p in
                    (value.q, value.scales, value.biases) if p is not None
                )
            return out
        finally:
            store.close()

    wide = sizes({"embed": 8, "attn": 8, "ffn": 8})
    narrow = sizes({"embed": 8, "attn": 8, "ffn": 4})

    assert narrow["blk.0.ffn_up.weight"] < wide["blk.0.ffn_up.weight"]
    assert narrow["token_embd.weight"] == wide["token_embd.weight"]
    assert narrow["blk.0.attn_q.weight"] == wide["blk.0.attn_q.weight"]


@pytest.mark.skipif(not SMOL.exists(), reason="SmolLM-135M database not present")
def test_a_mix_is_not_overruled_by_the_packed_index(tmp_path):
    """The index holds one width; a mix asks for several, and must win.

    This is the quiet failure the whole feature is exposed to. The index is
    the fast path, it is consulted before anything else, and it would have
    handed back 8-bit weights for a group the search had just decided should
    be 4-bit -- producing a table of measurements of a mix that was never
    loaded. The tensors whose role does match keep using the index, because
    there is nothing wrong with those.

    A single width keeps its old behaviour on purpose: there the index wins
    and `infer` says so, since rebuilding it per run is the cost it exists
    to remove.
    """
    import shutil

    from reminis import packed_index
    from reminis.backend import select as select_backend
    from reminis.weights import WeightStore

    backend = select_backend("inference")
    if not backend.can_pack():
        pytest.skip("this backend holds no packed weights")

    db = tmp_path / "indexed.db"
    shutil.copy(SMOL, db)
    packed_index.build(str(db), bits=8, group_size=128)

    names = ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_up.weight"]

    def load(mix):
        store = WeightStore(str(db), backend=backend, pack_bits=mix)
        try:
            value = {n: store.get(n) for n in names}
            ffn = value["blk.0.ffn_up.weight"]
            return store.from_index, sum(
                int(p.nbytes) for p in (ffn.q, ffn.scales, ffn.biases)
                if p is not None)
        finally:
            store.close()

    matching, wide_bytes = load({"embed": 8, "attn": 8, "ffn": 8})
    differing, narrow_bytes = load({"embed": 8, "attn": 8, "ffn": 4})

    assert matching == 3, "a mix at the index's own width should use it"
    assert differing == 2, "only the group that disagrees should bypass it"
    assert narrow_bytes < wide_bytes, "the 4-bit group was really packed at 4"

    store = WeightStore(str(db), backend=backend, pack_bits=4)
    try:
        store.get("blk.0.ffn_up.weight")
        assert store.from_index == 1
        assert store.index_bits == 8
    finally:
        store.close()
