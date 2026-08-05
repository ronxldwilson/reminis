"""Verify the viewer's architecture diagram describes the model it was given.

The diagram is built by JavaScript at page load, so checking the generated HTML
for a string proves nothing about what a reader ends up seeing. This test runs
the viewer's own script under node with a stub DOM and asserts on the diagram it
actually produces. Without node on PATH the test skips rather than pretending.

The cases that matter are the ones that used to be drawn wrong: a Mixture-of-
Experts model, whose file size says nothing about how much of it runs per token,
and the attention-free families (Mamba, RWKV), which used to render an empty
Feed-Forward box reading 0 B.
"""

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reminis.viewer import generate_viewer

TMP = Path(__file__).parent / "tmp_viewer"

# Runs the page's script with just enough DOM to capture the diagram, then
# flattens the result to text so assertions read like what a person would see.
HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const script = fs.readFileSync(process.argv[2], "utf8").match(/<script>([\s\S]*)<\/script>/)[1];
const store = {};
function el(id) {
  if (!store[id]) store[id] = {
    id, innerHTML: "", textContent: "", value: "", style: {}, dataset: {},
    classList: {add(){}, remove(){}, contains(){return false;}},
    addEventListener(){}, appendChild(){}, querySelectorAll(){return [];},
    getBoundingClientRect(){return {top:0,left:0,bottom:0,right:0,width:0,height:0};},
  };
  return store[id];
}
const document = {
  getElementById: el, querySelector: () => el("_q"), querySelectorAll: () => [],
  createElement: () => el("_c"), addEventListener(){}, body: el("_b"), documentElement: el("_h"),
};
const ctx = {
  document,
  window: {addEventListener(){}, matchMedia: () => ({matches:false, addEventListener(){}})},
  localStorage: {getItem: () => null, setItem(){}},
  console, requestAnimationFrame: f => f(), setTimeout,
};
ctx.window.document = document;
vm.createContext(ctx);
vm.runInContext(script, ctx, {filename: "viewer"});
console.log((store.archDiagram ? store.archDiagram.innerHTML : "(archDiagram never set)")
  .replace(/<span class="info-btn"[\s\S]*?<\/span>/g, "")
  .replace(/<[^>]+>/g, " ")
  .replace(/&middot;/g, "|").replace(/&mdash;/g, "--").replace(/&#39;/g, "'")
  .replace(/\s+/g, " "));
"""


def have_node() -> bool:
    return shutil.which("node") is not None


def render(db_path: Path, harness: Path) -> str:
    """The architecture diagram, as flattened text."""
    html = TMP / (db_path.stem + ".html")
    generate_viewer(str(db_path), str(html), verbose=False)
    out = subprocess.run(
        ["node", str(harness), str(html)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def build_db(path: Path, meta: dict, tensors: list) -> None:
    """A minimal reminis database. Only the diagram's inputs need to be real."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE model_meta (key TEXT PRIMARY KEY, value TEXT, type TEXT)"
    )
    conn.execute(
        "CREATE TABLE tensors (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, "
        "shape TEXT, dtype TEXT, dtype_id INTEGER, n_elements INTEGER, "
        "n_bytes INTEGER, data BLOB)"
    )
    for k, v in meta.items():
        conn.execute("INSERT INTO model_meta VALUES (?,?,?)", (k, str(v), "str"))
    for name, shape in tensors:
        n = 1
        for d in shape:
            n *= d
        # F32 payloads keep the stats and heatmap code paths on their real
        # branch rather than the quantized shortcut.
        conn.execute(
            "INSERT INTO tensors (name, shape, dtype, dtype_id, n_elements, n_bytes, data) "
            "VALUES (?,?,?,?,?,?,?)",
            (name, json.dumps(list(shape)), "F32", 0, n, n * 4, b"\x00\x00\x80\x3f" * n),
        )
    conn.commit()
    conn.close()


def moe_db(path: Path) -> None:
    """A GGUF-named MoE model: 8 experts, 2 used, stacked into 3-D tensors."""
    tensors = [("token_embd.weight", (64, 100))]
    for layer in range(4):
        for part in ("attn_q", "attn_k", "attn_v", "attn_output"):
            tensors.append((f"blk.{layer}.{part}.weight", (64, 64)))
        tensors.append((f"blk.{layer}.ffn_gate_inp.weight", (64, 8)))
        for part in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            tensors.append((f"blk.{layer}.{part}.weight", (64, 128, 8)))
        tensors.append((f"blk.{layer}.attn_norm.weight", (64,)))
        tensors.append((f"blk.{layer}.ffn_norm.weight", (64,)))
    tensors.append(("output_norm.weight", (64,)))
    build_db(path, {
        "general.architecture": "toymoe",
        "general.name": "toy-moe",
        "toymoe.block_count": 4,
        "toymoe.expert_count": 8,
        "toymoe.expert_used_count": 2,
    }, tensors)


def mixtral_db(path: Path) -> None:
    """The same idea under PyTorch names, with experts stored one per tensor."""
    tensors = [("model.embed_tokens.weight", (64, 100))]
    for layer in range(2):
        for part in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors.append((f"model.layers.{layer}.self_attn.{part}.weight", (64, 64)))
        tensors.append((f"model.layers.{layer}.block_sparse_moe.gate.weight", (8, 64)))
        for expert in range(8):
            for part in ("w1", "w2", "w3"):
                tensors.append(
                    (f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{part}.weight", (64, 128))
                )
        tensors.append((f"model.layers.{layer}.input_layernorm.weight", (64,)))
    tensors.append(("lm_head.weight", (64, 100)))
    build_db(path, {
        "general.architecture": "mixtral",
        "general.name": "toy-mixtral",
        "config.num_local_experts": 8,
        "config.num_experts_per_tok": 2,
    }, tensors)


def mamba_db(path: Path) -> None:
    """A state-space model: `ssm_*` and a norm, no attention anywhere."""
    tensors = [("token_embd.weight", (64, 100))]
    for layer in range(4):
        for part in ("ssm_in", "ssm_x", "ssm_dt", "ssm_out"):
            tensors.append((f"blk.{layer}.{part}.weight", (64, 128)))
        tensors.append((f"blk.{layer}.ssm_conv1d.weight", (4, 128)))
        tensors.append((f"blk.{layer}.ssm_a", (16, 128)))
        tensors.append((f"blk.{layer}.ssm_d", (128,)))
        tensors.append((f"blk.{layer}.attn_norm.weight", (64,)))
    tensors.append(("output_norm.weight", (64,)))
    build_db(path, {
        "general.architecture": "mamba",
        "general.name": "toy-mamba",
        "mamba.block_count": 4,
    }, tensors)


def rwkv_db(path: Path) -> None:
    """Time mixing in place of attention, channel mixing in place of the FFN."""
    tensors = [("token_embd.weight", (64, 100))]
    for layer in range(4):
        for part in ("time_mix_key", "time_mix_value", "time_mix_receptance", "time_mix_output"):
            tensors.append((f"blk.{layer}.{part}.weight", (64, 64)))
        tensors.append((f"blk.{layer}.time_decay", (64,)))
        tensors.append((f"blk.{layer}.time_first", (64,)))
        for part in ("channel_mix_key", "channel_mix_value", "channel_mix_receptance"):
            tensors.append((f"blk.{layer}.{part}.weight", (64, 128)))
        tensors.append((f"blk.{layer}.attn_norm.weight", (64,)))
    tensors.append(("output_norm.weight", (64,)))
    build_db(path, {
        "general.architecture": "rwkv6",
        "general.name": "toy-rwkv",
    }, tensors)


def hybrid_db(path: Path) -> None:
    """A Granite-4.0-H shape: a state-space stack with attention every fifth
    block. Long enough that the diagram has to collapse it, which is where a
    fixed first-two/last-two window would hide every attention block."""
    tensors = [("token_embd.weight", (64, 100))]
    for layer in range(16):
        if layer % 5 == 4:
            for part in ("attn_q", "attn_k", "attn_v", "attn_output"):
                tensors.append((f"blk.{layer}.{part}.weight", (64, 64)))
        else:
            for part in ("ssm_in", "ssm_x", "ssm_dt", "ssm_out"):
                tensors.append((f"blk.{layer}.{part}.weight", (64, 128)))
        for part in ("ffn_gate", "ffn_up", "ffn_down"):
            tensors.append((f"blk.{layer}.{part}.weight", (64, 128)))
        tensors.append((f"blk.{layer}.attn_norm.weight", (64,)))
    tensors.append(("output_norm.weight", (64,)))
    build_db(path, {"general.architecture": "jamba", "general.name": "toy-hybrid"}, tensors)


def vision_db(path: Path) -> None:
    """A projector file: a vision tower and nothing else."""
    tensors = [("v.patch_embd.weight", (16, 768)), ("v.position_embd.weight", (64, 768))]
    for layer in range(3):
        for part in ("attn_q", "attn_k", "attn_v", "attn_out"):
            tensors.append((f"v.blk.{layer}.{part}.weight", (64, 64)))
        for part in ("ffn_up", "ffn_down"):
            tensors.append((f"v.blk.{layer}.{part}.weight", (64, 128)))
        tensors.append((f"v.blk.{layer}.ln1.weight", (64,)))
        tensors.append((f"v.blk.{layer}.ln2.weight", (64,)))
    tensors.append(("v.post_ln.weight", (64,)))
    tensors.append(("mm.model.fc.weight", (64, 256)))
    build_db(path, {"general.architecture": "clip", "general.name": "toy-projector"}, tensors)


def unknown_db(path: Path) -> None:
    """An architecture the viewer has never heard of."""
    tensors = [("token_embd.weight", (64, 100))]
    for layer in range(3):
        tensors.append((f"blk.{layer}.mystery_a.weight", (64, 64)))
        tensors.append((f"blk.{layer}.mystery_b.weight", (64, 128)))
    build_db(path, {"general.architecture": "whoknows", "general.name": "toy-unknown"}, tensors)


def dense_db(path: Path) -> None:
    """An ordinary transformer, to catch regressions in the common case."""
    tensors = [("token_embd.weight", (64, 100))]
    for layer in range(4):
        for part in ("attn_q", "attn_k", "attn_v", "attn_output"):
            tensors.append((f"blk.{layer}.{part}.weight", (64, 64)))
        for part in ("ffn_gate", "ffn_up", "ffn_down"):
            tensors.append((f"blk.{layer}.{part}.weight", (64, 128)))
        tensors.append((f"blk.{layer}.attn_norm.weight", (64,)))
        tensors.append((f"blk.{layer}.ffn_norm.weight", (64,)))
    tensors.append(("output_norm.weight", (64,)))
    tensors.append(("output.weight", (64, 100)))
    build_db(path, {"general.architecture": "llama", "general.name": "toy-dense"}, tensors)


def check(label: str, diagram: str, present: list, absent: list) -> None:
    for s in present:
        assert s in diagram, f"{label}: expected {s!r} in the diagram\n---\n{diagram}"
    for s in absent:
        assert s not in diagram, f"{label}: did not expect {s!r} in the diagram\n---\n{diagram}"
    print(f"  {label}: ok")


def main() -> None:
    if not have_node():
        print("SKIPPED - node is not on PATH, so the diagram cannot be rendered")
        return

    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    harness = TMP / "harness.js"
    harness.write_text(HARNESS)

    print("Architecture diagram")

    moe_db(TMP / "moe.db")
    d = render(TMP / "moe.db", harness)
    check("MoE (GGUF names)", d, present=[
        "Mixture of Experts",
        "8 experts | 2 active per token",
        "Experts",
        "8 experts, 2 active per token",
        "Router",
        "run for any one token",
        "router tensors",
        "Self-Attention",
    ], absent=["Feed-Forward"])
    # Attention comes before the experts, in the order data flows through a block.
    assert d.index("Self-Attention") < d.index("Router") < d.index("8 experts, 2 active"), \
        "block boxes are not in the order data flows through them"
    print("  MoE: attention, then the router, then the experts")

    mixtral_db(TMP / "mixtral.db")
    d = render(TMP / "mixtral.db", harness)
    # Expert count comes from the HF config key here, and the router is named
    # `block_sparse_moe.gate` rather than `ffn_gate_inp`.
    check("MoE (PyTorch names)", d, present=[
        "8 experts | 2 active per token",
        "router tensor",
        "Token Embedding",
        "Output Projection",
    ], absent=["0 router tensors"])

    mamba_db(TMP / "mamba.db")
    d = render(TMP / "mamba.db", harness)
    check("Mamba", d, present=["state-space block", "State Space (SSM)"],
          absent=["Self-Attention", "Feed-Forward", "transformer block"])

    rwkv_db(TMP / "rwkv.db")
    d = render(TMP / "rwkv.db", harness)
    check("RWKV", d, present=["RWKV block", "Time Mixing", "Channel Mixing"],
          absent=["Self-Attention", "transformer block"])

    hybrid_db(TMP / "hybrid.db")
    d = render(TMP / "hybrid.db", harness)
    check("Hybrid", d, present=[
        "hybrid block", "Self-Attention", "State Space (SSM)", "Feed-Forward",
        # The attention blocks are at 4, 9 and 14, all of them inside what a
        # first-two/last-two collapse would drop.
        "Block 4", "Block 9", "Block 14",
        "identical blocks, through block",
    ], absent=["transformer block", "same structure"])

    unknown_db(TMP / "unknown.db")
    d = render(TMP / "unknown.db", harness)
    check("Unknown architecture", d, present=["Block weights", "mystery_a.weight"],
          absent=["Self-Attention", "Feed-Forward", "transformer block"])

    vision_db(TMP / "vision.db")
    d = render(TMP / "vision.db", harness)
    # Everything in the file is accounted for, including the projector, which
    # is the only reason a projector file exists.
    check("Vision tower", d, present=[
        "Vision encoder", "Patch embedding", "Position embedding",
        "3 vision blocks", "Self-Attention", "Feed-Forward",
        "Output normalization", "Projector",
        "no text transformer blocks",
    ], absent=[])

    dense_db(TMP / "dense.db")
    d = render(TMP / "dense.db", harness)
    check("Dense transformer", d, present=[
        "transformer block", "Self-Attention", "Feed-Forward",
        "Token Embedding", "Output Projection",
    ], absent=["Mixture of Experts", "State Space", "Block weights"])
    assert d.index("Self-Attention") < d.index("Feed-Forward"), \
        "feed-forward drawn before the attention it follows"

    # Real files, where the local ones exist. Synthetic databases prove the
    # naming logic; only a real model proves it against a real file.
    models = Path(__file__).parent.parent / "models"
    real = [
        ("granite-3.1-1b-a400m-instruct-Q4_K_M.db", ["Mixture of Experts", "32 experts | 8 active per token"]),
        ("mamba-130m.db", ["state-space block", "State Space (SSM)"]),
        ("rwkv7-1.5b.db", ["RWKV block", "Time Mixing", "Channel Mixing"]),
        ("SmolLM-135M.f16.db", ["transformer block", "Self-Attention", "Feed-Forward"]),
        ("smolvlm-mmproj.db", ["Vision encoder", "Patch embedding", "12 vision blocks", "Projector"]),
        # 36 state-space blocks with attention at 5, 15, 25 and 35.
        ("granite4-h-micro.db", ["hybrid block", "State Space (SSM)", "Self-Attention",
                                 "Block 5", "Block 15", "Block 25", "Block 35"]),
    ]
    for filename, expected in real:
        db = models / filename
        if not db.exists():
            print(f"  {filename}: not present locally, skipped")
            continue
        check(filename, render(db, harness), present=expected, absent=[])

    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 78)
    print("ALL VIEWER TESTS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
