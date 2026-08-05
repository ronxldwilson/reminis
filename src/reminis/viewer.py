"""Generate a self-contained HTML viewer for a reminis database."""

import json
import sqlite3
import struct
import webbrowser
from pathlib import Path

import numpy as np


def _compute_tensor_stats(data_blob: bytes, dtype_name: str, n_elements: int) -> dict:
    """Compute basic statistics for a tensor."""
    try:
        if dtype_name == "F32":
            arr = np.frombuffer(data_blob, dtype=np.float32)
        elif dtype_name in ("F16", "BF16"):
            arr = np.frombuffer(data_blob, dtype=np.float16)
        else:
            return {"quantized": True, "n_bytes": len(data_blob)}

        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "abs_mean": float(np.mean(np.abs(arr))),
            "zeros_pct": float(np.sum(arr == 0) / len(arr) * 100),
            "n_elements": len(arr),
        }
    except Exception:
        return {"error": True, "n_bytes": len(data_blob)}


def _sample_heatmap(data_blob: bytes, dtype_name: str, shape: list, size: int = 48) -> list | None:
    """Sample a 2D heatmap from tensor data."""
    try:
        if dtype_name == "F32":
            arr = np.frombuffer(data_blob, dtype=np.float32)
        elif dtype_name in ("F16", "BF16"):
            arr = np.frombuffer(data_blob, dtype=np.float16).astype(np.float32)
        else:
            return None

        if len(shape) < 2:
            return None

        rows, cols = shape[-2], shape[-1]
        arr = arr[:rows * cols].reshape(rows, cols)

        row_idx = np.linspace(0, rows - 1, min(size, rows), dtype=int)
        col_idx = np.linspace(0, cols - 1, min(size, cols), dtype=int)
        sampled = arr[np.ix_(row_idx, col_idx)]

        vmax = float(np.max(np.abs(sampled)))
        if vmax > 0:
            sampled = sampled / vmax

        return [[round(float(v), 3) for v in row] for row in sampled]
    except Exception:
        return None


def generate_viewer(db_path: str, output_path: str | None = None, verbose: bool = True) -> str:
    """Generate a self-contained HTML viewer for a reminis database."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if output_path is None:
        output_path = str(db_path.with_suffix(".html"))

    conn = sqlite3.connect(str(db_path))

    # Gather metadata
    meta = {}
    for key, value, type_name in conn.execute("SELECT key, value, type FROM model_meta"):
        meta[key] = {"value": value, "type": type_name}

    # Gather tensor info + stats
    tensors = []
    rows = conn.execute(
        "SELECT name, shape, dtype, dtype_id, n_elements, n_bytes, data FROM tensors ORDER BY id"
    ).fetchall()

    for name, shape_str, dtype_name, dtype_id, n_elements, n_bytes, data_blob in rows:
        shape = json.loads(shape_str)
        stats = _compute_tensor_stats(data_blob, dtype_name, n_elements)
        heatmap = _sample_heatmap(data_blob, dtype_name, shape)

        tensors.append({
            "name": name,
            "shape": shape,
            "dtype": dtype_name,
            "n_elements": n_elements,
            "n_bytes": n_bytes,
            "stats": stats,
            "heatmap": heatmap,
        })

    conn.close()

    # Summary stats
    total_params = sum(t["n_elements"] for t in tensors)
    total_bytes = sum(t["n_bytes"] for t in tensors)
    dtype_counts = {}
    for t in tensors:
        d = t["dtype"]
        if d not in dtype_counts:
            dtype_counts[d] = {"count": 0, "bytes": 0, "params": 0}
        dtype_counts[d]["count"] += 1
        dtype_counts[d]["bytes"] += t["n_bytes"]
        dtype_counts[d]["params"] += t["n_elements"]

    model_name = meta.get("general.name", {}).get("value", db_path.stem)
    arch = meta.get("general.architecture", {}).get("value", "unknown")

    viewer_data = {
        "model_name": model_name,
        "architecture": arch,
        "db_file": db_path.name,
        "db_size_mb": round(db_path.stat().st_size / (1024 * 1024), 1),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "dtype_counts": dtype_counts,
        "meta": {k: v["value"] for k, v in meta.items()},
        "tensors": tensors,
    }

    html = _build_html(viewer_data)

    with open(output_path, "w") as f:
        f.write(html)

    if verbose:
        size_mb = len(html) / (1024 * 1024)
        print(f"Viewer written to {output_path} ({size_mb:.1f} MB)")

    return output_path


def _build_html(data: dict) -> str:
    data_json = json.dumps(data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{data["model_name"]} - reminis viewer</title>
<style>
:root {{
  --bg: #0e1117;
  --bg2: #161b22;
  --bg3: #1c2129;
  --border: #2d333b;
  --text: #e6edf3;
  --text2: #8b949e;
  --accent: #58a6ff;
  --accent2: #1f6feb;
  --green: #3fb950;
  --orange: #d29922;
  --red: #f85149;
  --mono: "SF Mono", "Cascadia Code", "Fira Code", Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}}
@media (prefers-color-scheme: light) {{
  :root {{
    --bg: #ffffff;
    --bg2: #f6f8fa;
    --bg3: #ebeef1;
    --border: #d0d7de;
    --text: #1f2328;
    --text2: #656d76;
    --accent: #0969da;
    --accent2: #0550ae;
    --green: #1a7f37;
    --orange: #9a6700;
    --red: #cf222e;
  }}
}}
:root[data-theme="light"] {{
  --bg: #ffffff;
  --bg2: #f6f8fa;
  --bg3: #ebeef1;
  --border: #d0d7de;
  --text: #1f2328;
  --text2: #656d76;
  --accent: #0969da;
  --accent2: #0550ae;
  --green: #1a7f37;
  --orange: #9a6700;
  --red: #cf222e;
}}
:root[data-theme="dark"] {{
  --bg: #0e1117;
  --bg2: #161b22;
  --bg3: #1c2129;
  --border: #2d333b;
  --text: #e6edf3;
  --text2: #8b949e;
  --accent: #58a6ff;
  --accent2: #1f6feb;
  --green: #3fb950;
  --orange: #d29922;
  --red: #f85149;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}

/* Header */
.header {{
  padding: 32px 0 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}}
.header h1 {{
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 4px;
}}
.header .sub {{
  color: var(--text2);
  font-size: 14px;
  font-family: var(--mono);
}}

/* Stat cards */
.stats {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}}
.stat {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}}
.stat .label {{
  font-size: 12px;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
}}
.stat .value {{
  font-size: 24px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}}
.stat .detail {{
  font-size: 12px;
  color: var(--text2);
  margin-top: 2px;
}}

/* Tabs */
.tabs {{
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}}
.tab {{
  padding: 10px 20px;
  cursor: pointer;
  color: var(--text2);
  font-size: 14px;
  font-weight: 500;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
  background: none;
  border-top: none;
  border-left: none;
  border-right: none;
  font-family: var(--sans);
}}
.tab:hover {{ color: var(--text); }}
.tab.active {{
  color: var(--accent);
  border-bottom-color: var(--accent);
}}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* Search & filters */
.toolbar {{
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}}
.toolbar input, .toolbar select {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 12px;
  color: var(--text);
  font-size: 14px;
  font-family: var(--sans);
}}
.toolbar input {{ flex: 1; min-width: 200px; }}
.toolbar input:focus, .toolbar select:focus {{
  outline: none;
  border-color: var(--accent);
}}
.toolbar select {{ min-width: 120px; }}
.count {{
  color: var(--text2);
  font-size: 13px;
  padding: 8px 0;
  align-self: center;
}}

/* Tensor list */
.tensor {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 8px;
  overflow: hidden;
}}
.tensor-header {{
  display: grid;
  grid-template-columns: 1fr auto auto auto;
  gap: 16px;
  padding: 12px 16px;
  cursor: pointer;
  align-items: center;
  transition: background 0.1s;
}}
.tensor-header:hover {{ background: var(--bg3); }}
.tensor-name {{
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.tensor-meta {{
  font-size: 12px;
  color: var(--text2);
  font-family: var(--mono);
  white-space: nowrap;
}}
.tensor-badge {{
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 500;
  background: var(--bg3);
  color: var(--text2);
  border: 1px solid var(--border);
}}
.tensor-detail {{
  display: none;
  padding: 0 16px 16px;
  border-top: 1px solid var(--border);
}}
.tensor.open .tensor-detail {{ display: block; }}
.detail-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px;
  margin-top: 12px;
}}
.detail-item {{
  background: var(--bg3);
  border-radius: 6px;
  padding: 8px 12px;
}}
.detail-item .dlabel {{
  font-size: 11px;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.detail-item .dvalue {{
  font-size: 16px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  font-family: var(--mono);
}}

/* Heatmap */
.heatmap-wrap {{
  margin-top: 12px;
  overflow-x: auto;
}}
.heatmap-wrap canvas {{
  border-radius: 4px;
  image-rendering: pixelated;
  width: 100%;
  max-width: 500px;
  height: auto;
}}

/* Metadata table */
.meta-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
.meta-table th, .meta-table td {{
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}}
.meta-table th {{
  color: var(--text2);
  font-weight: 500;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  background: var(--bg2);
}}
.meta-table td:first-child {{
  font-family: var(--mono);
  color: var(--accent);
  white-space: nowrap;
}}
.meta-table td:last-child {{
  font-family: var(--mono);
  word-break: break-all;
}}

/* Dtype breakdown bar */
.dtype-bar {{
  display: flex;
  height: 32px;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 16px;
  border: 1px solid var(--border);
}}
.dtype-segment {{
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 600;
  color: #fff;
  min-width: 40px;
  transition: opacity 0.15s;
  cursor: default;
}}
.dtype-segment:hover {{ opacity: 0.85; }}
.dtype-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 20px;
  font-size: 12px;
}}
.dtype-legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
}}
.dtype-dot {{
  width: 10px;
  height: 10px;
  border-radius: 3px;
}}

/* Info buttons */
.info-btn {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg3);
  color: var(--text2);
  font-size: 11px;
  font-weight: 600;
  font-style: italic;
  font-family: Georgia, serif;
  cursor: pointer;
  margin-left: 6px;
  vertical-align: middle;
  transition: border-color 0.15s, color 0.15s;
  flex-shrink: 0;
  position: relative;
}}
.info-btn:hover {{
  border-color: var(--accent);
  color: var(--accent);
}}
.info-popup {{
  display: none;
  position: fixed;
  z-index: 1000;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 20px;
  max-width: 380px;
  width: calc(100vw - 40px);
  box-shadow: 0 8px 30px rgba(0,0,0,0.4);
  font-size: 13px;
  line-height: 1.6;
  color: var(--text);
}}
.info-popup.visible {{ display: block; }}
.info-popup .info-title {{
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 6px;
  color: var(--accent);
}}
.info-popup .info-body {{
  color: var(--text2);
}}
.info-popup .info-body strong {{
  color: var(--text);
  font-weight: 500;
}}
.info-overlay {{
  display: none;
  position: fixed;
  inset: 0;
  z-index: 999;
}}
.info-overlay.visible {{ display: block; }}

/* Intro banner */
.intro-banner {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 24px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--text2);
  display: flex;
  gap: 12px;
  align-items: flex-start;
}}
.intro-banner .intro-icon {{
  font-size: 20px;
  flex-shrink: 0;
  margin-top: 2px;
}}
.intro-banner .intro-text strong {{
  color: var(--text);
}}
.intro-dismiss {{
  margin-left: auto;
  background: none;
  border: none;
  color: var(--text2);
  cursor: pointer;
  font-size: 18px;
  padding: 0 4px;
  flex-shrink: 0;
}}
.intro-dismiss:hover {{ color: var(--text); }}

/* Footer */
.footer {{
  margin-top: 40px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  color: var(--text2);
  font-size: 12px;
  text-align: center;
}}
.footer a {{ color: var(--accent); text-decoration: none; }}
.footer a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1 id="modelName"></h1>
  <div class="sub" id="headerSub"></div>
</div>

<div class="intro-banner" id="introBanner">
  <span class="intro-icon">&#128218;</span>
  <div class="intro-text">
    <strong>You're looking inside a neural network.</strong>
    An LLM is made of millions of numbers (weights) organized into layers. Each layer transforms text step by step.
    This viewer lets you browse every weight tensor stored in a SQLite database. Click the
    <span class="info-btn" style="display:inline-flex;cursor:default;margin:0 2px;">i</span>
    buttons throughout the page to learn what each part means.
  </div>
  <button class="intro-dismiss" onclick="document.getElementById('introBanner').style.display='none'">&times;</button>
</div>

<div class="stats" id="statsGrid"></div>

<div id="dtypeViz"></div>

<div class="tabs">
  <button class="tab active" data-tab="architecture">Architecture</button>
  <button class="tab" data-tab="tensors">Tensors</button>
  <button class="tab" data-tab="metadata">Metadata</button>
</div>

<div class="tab-content active" id="tab-architecture">
  <div id="archDiagram"></div>
</div>

<div class="tab-content" id="tab-tensors">
  <div class="toolbar">
    <input type="text" id="search" placeholder="Search tensors...">
    <select id="dtypeFilter"><option value="">All types</option></select>
    <span class="count" id="tensorCount"></span>
  </div>
  <div id="tensorList"></div>
</div>

<div class="tab-content" id="tab-metadata">
  <p style="color:var(--text2);font-size:13px;margin-bottom:12px;">
    These are the configuration values stored inside the model file &mdash; they tell inference engines how to load and run the model.
    <span class="info-btn" onclick="showInfo(this,'Metadata','These key-value pairs describe the model architecture, tokenizer settings, and training configuration. They are read by tools like llama.cpp to know how to interpret the weight data. For example, context_length tells the maximum number of tokens the model can process at once.')">i</span>
  </p>
  <table class="meta-table" id="metaTable">
    <thead><tr><th>Key</th><th>Value</th></tr></thead>
    <tbody id="metaBody"></tbody>
  </table>
</div>

<div class="footer">
  Generated by <a href="https://github.com/ronxldwilson/reminis">reminis</a>
</div>

</div>

<div class="info-overlay" id="infoOverlay" onclick="hideInfo()"></div>
<div class="info-popup" id="infoPopup">
  <div class="info-title" id="infoTitle"></div>
  <div class="info-body" id="infoBody"></div>
</div>

<script>
const DATA = {data_json};

const DTYPE_COLORS = {{
  F32: "#58a6ff", F16: "#3fb950", BF16: "#a371f7",
  Q2_K: "#f0883e", Q3_K: "#d29922", Q4_K: "#f85149",
  Q5_K: "#db61a2", Q5_0: "#bc8cff", Q5_1: "#79c0ff",
  Q6_K: "#56d364", Q8_0: "#ff7b72", Q8_K: "#ffa657",
  IQ1_S: "#7ee787", IQ2_S: "#a5d6ff", IQ2_XS: "#d2a8ff",
  IQ3_S: "#ffd700", IQ3_M: "#e3b341", IQ3_XXS: "#b392f0",
  IQ4_NL: "#f9826c", IQ4_XS: "#ffab70",
}};
function dtypeColor(d) {{ return DTYPE_COLORS[d] || "#8b949e"; }}

function fmtNum(n) {{
  if (n >= 1e9) return (n/1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n/1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n/1e3).toFixed(1) + "K";
  return n.toString();
}}
function fmtBytes(b) {{
  if (b >= 1024*1024*1024) return (b/1024/1024/1024).toFixed(1) + " GB";
  if (b >= 1024*1024) return (b/1024/1024).toFixed(1) + " MB";
  if (b >= 1024) return (b/1024).toFixed(1) + " KB";
  return b + " B";
}}
function fmtSci(v) {{
  if (v === undefined || v === null) return "-";
  if (Math.abs(v) < 0.001 || Math.abs(v) >= 10000) return v.toExponential(3);
  return v.toFixed(4);
}}

// Info popup system
function showInfo(btn, title, body) {{
  const popup = document.getElementById("infoPopup");
  const overlay = document.getElementById("infoOverlay");
  document.getElementById("infoTitle").textContent = title;
  document.getElementById("infoBody").innerHTML = body;
  const rect = btn.getBoundingClientRect();
  let top = rect.bottom + 8;
  let left = rect.left;
  if (left + 380 > window.innerWidth) left = window.innerWidth - 400;
  if (left < 20) left = 20;
  if (top + 200 > window.innerHeight) top = rect.top - 200;
  popup.style.top = top + "px";
  popup.style.left = left + "px";
  popup.classList.add("visible");
  overlay.classList.add("visible");
}}
function hideInfo() {{
  document.getElementById("infoPopup").classList.remove("visible");
  document.getElementById("infoOverlay").classList.remove("visible");
}}

// Tensor name explainer
const TENSOR_EXPLANATIONS = {{
  "token_embd": "The <strong>embedding layer</strong> converts each word/token into a vector of numbers the model can work with. It is essentially a lookup table: token ID in, vector out.",
  "output_norm": "The <strong>final normalization</strong> layer that stabilizes values right before the model predicts the next token. Without it, numbers could drift to extreme ranges.",
  "output": "The <strong>output projection</strong> converts the model's internal representation back into a probability for each possible next token in the vocabulary.",
  "attn_q": "The <strong>query (Q) matrix</strong> in the attention mechanism. It represents 'what am I looking for?' for each token. Multiplied with keys to compute attention scores.",
  "attn_k": "The <strong>key (K) matrix</strong> in attention. It represents 'what do I contain?' for each token. Other tokens compare their queries against these keys to decide what to attend to.",
  "attn_v": "The <strong>value (V) matrix</strong> in attention. Once attention scores decide which tokens are relevant, the values are what actually gets passed forward. It is the 'content' that gets read.",
  "attn_output": "The <strong>attention output projection</strong>. After attention combines the value vectors, this matrix projects the result back to the model's main dimension.",
  "attn_out": "The <strong>attention output projection</strong>. After attention combines the value vectors, this matrix projects the result back to the model's main dimension. (Vision encoders spell this <code>attn_out</code> rather than <code>attn_output</code>.)",
  "ffn_gate_exps": "The <strong>expert gate projections</strong> in a Mixture-of-Experts layer. This tensor stacks the gate matrices for every expert; a router picks which few experts each token actually uses, so most of these weights sit idle on any given token.",
  "ffn_up_exps": "The <strong>expert up projections</strong> in a Mixture-of-Experts layer, stacked across all experts.",
  "ffn_down_exps": "The <strong>expert down projections</strong> in a Mixture-of-Experts layer, stacked across all experts.",
  "ffn_gate_inp": "The <strong>router</strong> of a Mixture-of-Experts layer. It scores each expert per token and selects the top few to run, which is what makes MoE cheap at inference despite the large total parameter count.",
  "attn_norm": "The <strong>attention layer norm</strong> normalizes values before they enter the attention mechanism. This keeps training stable and helps the model learn.",
  "ffn_gate": "The <strong>gate projection</strong> in the feed-forward network (FFN). In SwiGLU architectures, it controls how much signal passes through, acting like a learned filter.",
  "ffn_up": "The <strong>up projection</strong> expands the representation to a wider dimension inside the FFN. The model thinks in a bigger space here before compressing back down.",
  "ffn_down": "The <strong>down projection</strong> compresses the FFN's wider representation back to the model's main dimension. Information is distilled on the way through.",
  "ffn_norm": "The <strong>FFN layer norm</strong> normalizes values before the feed-forward network. Same purpose as attn_norm but for the FFN sub-layer.",

  // State-space (Mamba) tensors. These models have no attention at all: the
  // block mixes across time with a recurrent scan instead.
  "ssm_in": "The <strong>input projection</strong> of a state-space (Mamba) block. It expands the token representation into the wider inner dimension the recurrent scan runs in.",
  "ssm_conv1d": "The <strong>short causal convolution</strong> in a Mamba block. Before the recurrent scan, each channel is mixed over a small window of recent tokens &mdash; a cheap local-context pass that attention would otherwise have to pay for.",
  "ssm_x": "The <strong>selection projection</strong> in a Mamba block. It reads the current token and produces the per-token values of <code>B</code>, <code>C</code>, and the timestep, which is what makes the state space <em>selective</em> rather than fixed.",
  "ssm_dt": "The <strong>timestep (delta) projection</strong> in a Mamba block. It decides, per token and per channel, how much the recurrent state should be updated versus carried forward &mdash; effectively a learned forget gate.",
  "ssm_a": "The <strong>state transition matrix</strong> <code>A</code> of a Mamba block, stored in log form. It governs how the hidden state decays as the scan moves forward, so it controls how far back information survives.",
  "ssm_d": "The <strong>skip (D) parameter</strong> of a Mamba block: a direct per-channel path from input to output that bypasses the recurrent state, much like a residual connection inside the scan.",
  "ssm_out": "The <strong>output projection</strong> of a state-space block. It projects the scan's result back down to the model's main dimension.",
  "ssm_norm": "The <strong>normalization inside the state-space block</strong>, applied to the scan output before it is projected back down.",

  // RWKV tensors. Another attention-free design: time mixing carries
  // information across tokens, channel mixing plays the role of the FFN.
  "time_mix": "Part of RWKV's <strong>time mixing</strong> block, which is what replaces attention. Instead of comparing every token against every other, it carries a running state forward with a learned per-channel decay, so cost stays constant per token.",
  "time_decay": "The <strong>decay rates</strong> in an RWKV time-mixing block. Each channel forgets the past at its own learned speed, which is how the model keeps some information for a long time and other information only briefly.",
  "time_first": "The <strong>current-token bonus</strong> in RWKV time mixing. It gives the token being processed right now extra weight relative to the decayed history.",
  "channel_mix": "RWKV's <strong>channel mixing</strong> block. It plays the same role the feed-forward network plays in a transformer: transform each position's representation, with no mixing across positions.",
  "token_shift": "The <strong>token shift</strong> in RWKV. Each block blends the current token's vector with the previous one before mixing, giving the model a cheap one-step look backwards.",

  // PyTorch / safetensors spellings of the same tensors. A model imported from
  // safetensors uses these names throughout, so without them every tensor
  // would fall back to the generic explanation.
  "embed_tokens": "The <strong>embedding layer</strong> converts each word/token into a vector of numbers the model can work with. It is essentially a lookup table: token ID in, vector out. (GGUF calls this <code>token_embd</code>.)",
  "lm_head": "The <strong>output projection</strong> converts the model's internal representation back into a probability for each possible next token in the vocabulary. Many models tie this to the embedding layer instead of storing it separately. (GGUF calls this <code>output</code>.)",
  "q_proj": "The <strong>query (Q) matrix</strong> in the attention mechanism. It represents 'what am I looking for?' for each token. Multiplied with keys to compute attention scores. (GGUF calls this <code>attn_q</code>.)",
  "k_proj": "The <strong>key (K) matrix</strong> in attention. It represents 'what do I contain?' for each token. Other tokens compare their queries against these keys to decide what to attend to. (GGUF calls this <code>attn_k</code>.)",
  "v_proj": "The <strong>value (V) matrix</strong> in attention. Once attention scores decide which tokens are relevant, the values are what actually gets passed forward. (GGUF calls this <code>attn_v</code>.)",
  "o_proj": "The <strong>attention output projection</strong>. After attention combines the value vectors, this matrix projects the result back to the model's main dimension. (GGUF calls this <code>attn_output</code>.)",
  "gate_proj": "The <strong>gate projection</strong> in the feed-forward network. In SwiGLU architectures, it controls how much signal passes through, acting like a learned filter. (GGUF calls this <code>ffn_gate</code>.)",
  "up_proj": "The <strong>up projection</strong> expands the representation to a wider dimension inside the FFN. The model thinks in a bigger space here before compressing back down. (GGUF calls this <code>ffn_up</code>.)",
  "down_proj": "The <strong>down projection</strong> compresses the FFN's wider representation back to the model's main dimension. (GGUF calls this <code>ffn_down</code>.)",
  "input_layernorm": "The <strong>attention layer norm</strong> normalizes values before they enter the attention mechanism. This keeps training stable and helps the model learn. (GGUF calls this <code>attn_norm</code>.)",
  "post_attention_layernorm": "The <strong>FFN layer norm</strong> normalizes values after attention and before the feed-forward network. (GGUF calls this <code>ffn_norm</code>.)",
  "lora_A": "One half of a <strong>LoRA factor pair</strong>. A LoRA fine-tune never touches the base weight; it learns two small matrices whose product is added to it. <code>lora_A</code> projects down to the small rank dimension.",
  "lora_B": "The other half of a <strong>LoRA factor pair</strong>. <code>lora_B</code> projects back up from the rank dimension, so <code>B @ A</code> has the same shape as the weight it modifies while holding far fewer numbers.",
}};

// Longest match wins, so `ffn_gate_exps` is not shadowed by `ffn_gate` and
// `attn_output` is not shadowed by `attn_out`. Relying on object insertion
// order here would be fragile.
function getTensorExplanation(name) {{
  let best = null;
  for (const [key, explanation] of Object.entries(TENSOR_EXPLANATIONS)) {{
    if (name.includes(key) && (best === null || key.length > best[0].length)) {{
      best = [key, explanation];
    }}
  }}
  if (best) return best[1];
  return "A weight tensor in the model. Each tensor is a multi-dimensional array of numbers that the model learned during training.";
}}

// Which stack a tensor belongs to, and its index within it.
//
// The prefix matters: vision-tower tensors are named `v.blk.N.*`, so an
// unanchored /blk\\.(\d+)/ matches the `blk.0` inside `v.blk.0` and folds
// vision layers into the text-block numbering. Anchoring on the prefix keeps
// the two stacks apart.
// Safetensors models keep PyTorch's names (`model.layers.0.self_attn.q_proj`)
// rather than GGUF's (`blk.0.attn_q`), so both conventions are matched here.
// Without this the diagram is simply empty for anything imported from
// safetensors.
function getBlockInfo(name) {{
  let m = name.match(/^blk\\.(\d+)\\./);
  if (m) return {{ stack: "text", idx: parseInt(m[1]) }};
  m = name.match(/^v\\.blk\\.(\d+)\\./);
  if (m) return {{ stack: "vision", idx: parseInt(m[1]) }};
  m = name.match(/^(?:model\\.)?layers\\.(\d+)\\./);
  if (m) return {{ stack: "text", idx: parseInt(m[1]) }};
  m = name.match(/vision_(?:tower|model)\\..*?layers\\.(\d+)\\./);
  if (m) return {{ stack: "vision", idx: parseInt(m[1]) }};
  return null;
}}

function getLayerFromName(name) {{
  const info = getBlockInfo(name);
  return info ? info.idx : -1;
}}

// Metadata keys are namespaced by architecture (`granitemoe.expert_count`,
// `mamba.ssm.state_size`), and the prefix is whatever `general.architecture`
// says. Matching on the suffix avoids having to know the family up front.
function metaNum(suffix) {{
  for (const [k, v] of Object.entries(DATA.meta)) {{
    if (k === suffix || k.endsWith("." + suffix)) {{
      const n = Number(v);
      if (Number.isFinite(n)) return n;
    }}
  }}
  return null;
}}

// What a tensor does inside its block.
//
// Hardcoding an attention box and an FFN box only describes transformers. A
// Mamba model has neither: every block is `ssm_*` plus one norm, so the old
// diagram drew a Feed-Forward box reading 0 B and an attention box holding
// nothing but a layer norm. Classifying by role instead lets each block show
// the parts it actually has.
//
// Order matters. `moe` is tested before `ffn` because expert tensors are also
// named `ffn_*`, and the recurrent roles are tested before `attn` because
// RWKV names its norms `attn_norm` despite having no attention.
const BLOCK_ROLES = [
  {{
    // The router is what makes a MoE model cheap, so it gets its own box
    // rather than disappearing into the experts it selects.
    key: "router", label: "Router", rgb: "241,161,79", order: 2,
    parts: "Picks the experts per token",
    test: n => /ffn_gate_inp|\\.gate\\.weight$|router/.test(n),
  }},
  {{
    key: "moe", label: "Experts", rgb: "188,140,255", order: 3,
    parts: "Stacked feed-forward experts",
    test: n => /(_exps)(\\.|$)|experts\\./.test(n),
  }},
  {{
    key: "ssm", label: "State Space (SSM)", rgb: "210,153,34", order: 1,
    parts: "in, conv1d, x, dt, A, D, out",
    test: n => /ssm_|conv1d|mamba/.test(n),
  }},
  {{
    key: "timemix", label: "Time Mixing", rgb: "210,153,34", order: 1,
    parts: "RWKV's replacement for attention",
    test: n => /time_mix|time_decay|time_first|time_faaaa|time_maa|token_shift/.test(n),
  }},
  {{
    key: "channelmix", label: "Channel Mixing", rgb: "63,185,80", order: 3,
    parts: "RWKV's feed-forward",
    test: n => /channel_mix/.test(n),
  }},
  {{
    key: "attn", label: "Self-Attention", rgb: "88,166,255", order: 1,
    parts: "Q, K, V, Output",
    test: n => /attn|attention/.test(n),
  }},
  {{
    key: "ffn", label: "Feed-Forward", rgb: "63,185,80", order: 3,
    parts: "Gate, Up, Down",
    test: n => /ffn_|mlp\\.|gate_proj|up_proj|down_proj/.test(n),
  }},
];

// Norms are counted in the block total but never get their own box: they are a
// few kilobytes next to megabytes, and a box for them would read as a stage of
// computation rather than the stabiliser it is.
function isNormTensor(name) {{
  // Vision towers spell their norms `ln1`/`ln2`/`post_ln` rather than `*_norm`.
  return /norm|layernorm|(^|[._])ln\d*([._]|$)/.test(name);
}}

function classifyBlockTensor(name) {{
  if (isNormTensor(name)) return "norm";
  for (const role of BLOCK_ROLES) {{
    if (role.test(name)) return role.key;
  }}
  return "other";
}}

// Mixture-of-Experts summary, or null for a dense model.
//
// The interesting number is not the file size: it is how much of the model runs
// for any one token. A router picks `expert_used_count` of `expert_count`
// experts, so the rest of the expert weights sit idle on every forward pass.
function getMoEInfo() {{
  const expertTensors = DATA.tensors.filter(t => /(_exps)(\\.|$)|experts\\./.test(t.name));
  if (expertTensors.length === 0) return null;

  const routers = DATA.tensors.filter(t => /ffn_gate_inp|\\.gate\\.weight$|router/.test(t.name));

  // Metadata is authoritative, but the key depends on where the model came
  // from: GGUF writes `<arch>.expert_count`, a HuggingFace config writes
  // `num_local_experts` or `n_routed_experts`.
  let total = metaNum("expert_count") || metaNum("num_local_experts")
    || metaNum("n_routed_experts") || metaNum("num_experts");
  if (!total) {{
    // GGUF stacks every expert into one 3-D tensor, so the count is its
    // trailing dimension. Safetensors keeps them separate and numbered, so
    // the count is the highest index seen.
    const s = expertTensors[0].shape;
    if (s.length === 3) {{
      total = s[s.length - 1];
    }} else {{
      let maxIdx = -1;
      expertTensors.forEach(t => {{
        const m = t.name.match(/experts\\.(\d+)\\./);
        if (m) maxIdx = Math.max(maxIdx, parseInt(m[1]));
      }});
      if (maxIdx >= 0) total = maxIdx + 1;
    }}
  }}
  const used = metaNum("expert_used_count") || metaNum("num_experts_per_tok")
    || metaNum("num_experts_per_token");

  const expertParams = expertTensors.reduce((a, t) => a + t.n_elements, 0);
  const expertBytes = expertTensors.reduce((a, t) => a + t.n_bytes, 0);
  const activeExpertParams = (total && used) ? Math.round(expertParams / total * used) : null;

  return {{
    total: total, used: used,
    routerCount: routers.length,
    routerBytes: routers.reduce((a, t) => a + t.n_bytes, 0),
    tensorCount: expertTensors.length,
    expertParams: expertParams,
    expertBytes: expertBytes,
    activeExpertParams: activeExpertParams,
    // Everything that is not an expert runs on every token.
    activeParams: activeExpertParams === null
      ? null
      : DATA.total_params - expertParams + activeExpertParams,
  }};
}}

// Header
document.getElementById("modelName").textContent = DATA.model_name;
document.getElementById("headerSub").textContent =
  DATA.architecture + " | " + DATA.db_file + " (" + DATA.db_size_mb + " MB)";

// Stats with info buttons
const STAT_INFO = {{
  "Parameters": "The total number of individual weight values in the model. Each parameter is a single number (like 0.0234 or -1.553) that was learned during training. More parameters generally means the model can store more knowledge, but also requires more memory.",
  "Tensors": "Weights are organized into <strong>tensors</strong> (multi-dimensional arrays of numbers). Think of a tensor as a spreadsheet: a 1D tensor is a row, a 2D tensor is a grid, etc. Each tensor has a specific job in the model, like computing attention or transforming representations.",
  "Weight Data": "The total size of all weight data in bytes. This is how much memory the model needs to store its learned knowledge. Quantized models use fewer bytes per parameter (e.g., 4-bit = 0.5 bytes) compared to full precision (float16 = 2 bytes, float32 = 4 bytes).",
  "Architecture": "The model family or design pattern. <strong>Llama</strong> is the most common architecture for open-source LLMs. It defines how layers are structured: self-attention followed by a feed-forward network (SwiGLU), with layer norms for stability.",
  "Database": "The SQLite database file that stores all the model weights and metadata as queryable rows. You can open it with any SQLite tool and run SQL queries against the model's weights.",
}};

const statsHTML = [
  ["Parameters", fmtNum(DATA.total_params), DATA.total_params.toLocaleString()],
  ["Tensors", DATA.tensors.length.toString(), ""],
  ["Weight Data", fmtBytes(DATA.total_bytes), ""],
  ["Architecture", DATA.architecture, ""],
  ["Database", DATA.db_size_mb + " MB", DATA.db_file],
].map(([label, value, detail]) => {{
  const info = STAT_INFO[label] || "";
  const iBtn = `<span class="info-btn" onclick="event.stopPropagation();showInfo(this,'${{label}}','${{info.replace(/'/g,"&#39;")}}')">i</span>`;
  return `<div class="stat"><div class="label">${{label}} ${{iBtn}}</div><div class="value">${{value}}</div>${{detail ? `<div class="detail">${{detail}}</div>` : ""}}</div>`;
}}).join("");
document.getElementById("statsGrid").innerHTML = statsHTML;

// Dtype breakdown with info
const dtypes = Object.entries(DATA.dtype_counts).sort((a,b) => b[1].bytes - a[1].bytes);
const dtypeInfo = "This bar shows how the model's weight data is distributed across <strong>data types</strong>. " +
  "<strong>F16</strong> (16-bit float) and <strong>F32</strong> (32-bit float) store full-precision values. " +
  "<strong>Q4_K, Q8_0</strong>, etc. are <strong>quantized</strong> types that compress weights to fewer bits (4-bit, 8-bit) " +
  "to save memory with minimal quality loss. A Q4 model uses roughly 4x less memory than F16.";
let barHTML = `<div style="display:flex;align-items:center;margin-bottom:8px;"><span style="font-size:13px;color:var(--text2);">Data type distribution</span><span class="info-btn" onclick="showInfo(this,'Data Types',\`${{dtypeInfo}}\`)">i</span></div>`;
barHTML += '<div class="dtype-bar">';
dtypes.forEach(([dtype, info]) => {{
  const pct = (info.bytes / DATA.total_bytes * 100);
  barHTML += `<div class="dtype-segment" style="width:${{Math.max(pct,2)}}%;background:${{dtypeColor(dtype)}}" title="${{dtype}}: ${{info.count}} tensors, ${{fmtBytes(info.bytes)}}">${{pct > 5 ? dtype : ""}}</div>`;
}});
barHTML += '</div><div class="dtype-legend">';
dtypes.forEach(([dtype, info]) => {{
  barHTML += `<div class="dtype-legend-item"><div class="dtype-dot" style="background:${{dtypeColor(dtype)}}"></div>${{dtype}}: ${{info.count}} (${{fmtBytes(info.bytes)}})</div>`;
}});
barHTML += '</div>';
document.getElementById("dtypeViz").innerHTML = barHTML;

// Dtype filter
const dtypeFilter = document.getElementById("dtypeFilter");
dtypes.forEach(([dtype]) => {{
  const opt = document.createElement("option");
  opt.value = dtype; opt.textContent = dtype;
  dtypeFilter.appendChild(opt);
}});

// Tabs
document.querySelectorAll(".tab").forEach(tab => {{
  tab.addEventListener("click", () => {{
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
  }});
}});

// Architecture diagram
function buildArchDiagram() {{
  const layers = {{}};
  const visionLayers = {{}};
  DATA.tensors.forEach(t => {{
    const info = getBlockInfo(t.name);
    if (!info) return;
    const target = info.stack === "vision" ? visionLayers : layers;
    if (!target[info.idx]) target[info.idx] = [];
    target[info.idx].push(t);
  }});
  const numLayers = Object.keys(layers).length;
  const numVision = Object.keys(visionLayers).length;
  // Both naming conventions again: GGUF's `token_embd`/`output_norm`/`output`
  // and PyTorch's `embed_tokens`/`model.norm`/`lm_head`.
  const embeds = DATA.tensors.filter(t => /token_embd|embed_tokens/.test(t.name));
  const norms = DATA.tensors.filter(t => /output_norm|^(?:model\.)?norm\.weight$/.test(t.name));
  const outputs = DATA.tensors.filter(t => t.name === "output.weight" || /lm_head/.test(t.name));

  // Not every stack is a transformer stack, so the wording follows what the
  // blocks actually contain rather than assuming attention is in there.
  const allBlockTensors = Object.values(layers).flat();
  const roleKeys = new Set(allBlockTensors.map(t => classifyBlockTensor(t.name)));
  const isSSM = roleKeys.has("ssm");
  const isRWKV = roleKeys.has("timemix") || roleKeys.has("channelmix");
  const hasAttn = roleKeys.has("attn") && allBlockTensors.some(
    t => classifyBlockTensor(t.name) === "attn" && !isNormTensor(t.name));
  let blockNoun = hasAttn ? "transformer block" : "block";
  if ((isSSM || isRWKV) && hasAttn) blockNoun = "hybrid block";
  else if (isSSM) blockNoun = "state-space block";
  else if (isRWKV) blockNoun = "RWKV block";

  const HOW_IT_WORKS = hasAttn
    ? "A large language model processes text by converting words into numbers (embeddings), then passing them through a stack of <strong>transformer blocks</strong>. Each block has two parts: <strong>self-attention</strong> (which lets every word look at every other word to understand context) and a <strong>feed-forward network</strong> (which transforms the representation). After all blocks, the output layer converts the final numbers back into a probability for each possible next word."
    : "This model has <strong>no attention layers</strong>. Instead of letting every token look at every other token &mdash; which costs more the longer the text gets &mdash; it carries a fixed-size <strong>recurrent state</strong> forward as it reads, updating that state one token at a time. The cost per token stays flat no matter how long the context is, at the price of having to compress the past into that state rather than being able to look it up. Everything else is familiar: embeddings in, a stack of blocks, an output projection back to vocabulary probabilities.";

  let html = `<div style="max-width:700px;margin:0 auto;">`;

  const intro = numLayers > 0
    ? `This shows how the model's layers are stacked. Text flows from top (input) to bottom (output), passing through each ${{blockNoun}}.`
    : "This file's components are shown below.";
  html += `<p style="color:var(--text2);font-size:13px;margin-bottom:16px;">
    ${{intro}}
    <span class="info-btn" onclick="showInfo(this,'How this model works','${{HOW_IT_WORKS.replace(/'/g,"&#39;")}}')">i</span>
  </p>`;

  // Mixture-of-Experts summary. Parameter count and working set are the same
  // number for a dense model and very different ones here, so a file size on
  // its own would overstate what actually runs per token.
  const moe = getMoEInfo();
  if (moe && moe.total) {{
    const routed = moe.used ? `${{moe.total}} experts &middot; ${{moe.used}} active per token` : `${{moe.total}} experts`;
    let activeLine = "";
    if (moe.activeParams !== null) {{
      const pct = (100 * moe.activeParams / DATA.total_params).toFixed(0);
      activeLine = `<div style="font-size:12px;color:var(--text2);margin-top:6px;">
        <strong style="color:var(--text);">${{fmtNum(moe.activeParams)}}</strong> of ${{fmtNum(DATA.total_params)}} parameters run for any one token (${{pct}}%).
        The other ${{fmtNum(DATA.total_params - moe.activeParams)}} sit in experts the router did not pick.
      </div>`;
    }}
    html += `<div style="border:1px solid rgba(188,140,255,0.35);border-radius:8px;padding:16px;margin-bottom:20px;background:rgba(188,140,255,0.07);">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
        <span style="font-weight:600;">Mixture of Experts
          <span class="info-btn" onclick="showInfo(this,'Mixture of Experts','Instead of one feed-forward network per block, this model holds <strong>${{moe.total}}</strong> of them, called experts. A small <strong>router</strong> (<code>ffn_gate_inp</code>) scores the experts for each token and runs only the top ${{moe.used || "few"}}. That is why a MoE model can hold a large number of parameters while costing far less than that to run: every token pays for the router, the attention layers, and its own handful of experts &mdash; not the whole file. The experts are stored stacked, so one tensor such as <code>ffn_gate_exps</code> holds every expert&#39;s gate matrix in a single 3-D array.')">i</span>
        </span>
        <span style="font-size:12px;color:var(--text2);font-family:var(--mono);">${{routed}} &middot; ${{fmtBytes(moe.expertBytes)}} of experts</span>
      </div>
      ${{activeLine}}
      ${{moe.routerCount > 0 ? `<div style="font-size:12px;color:var(--text2);margin-top:6px;">
        Routing is learned: ${{moe.routerCount}} router tensor${{moe.routerCount === 1 ? "" : "s"}} totalling ${{fmtBytes(moe.routerBytes)}} decide which experts each token reaches.
      </div>` : ""}}
    </div>`;
  }}

  // A vision tower is a separate stack. Rendering it inline with the text
  // blocks would imply text flows through it, which it does not.
  if (numVision > 0) {{
    const visTensors = Object.values(visionLayers).flat();
    const visBytes = visTensors.reduce((a, t) => a + t.n_bytes, 0);

    // The blocks are only part of the tower. A projector file's whole reason to
    // exist is the projector, and the image has to be cut into patches before
    // any block sees it — summarising the blocks alone leaves those out of a
    // diagram that claims to show the file's components.
    const patch = DATA.tensors.filter(t => /patch_emb/.test(t.name));
    const posEmbed = DATA.tensors.filter(t => /position_emb/.test(t.name));
    const projector = DATA.tensors.filter(t => /^mm\\.|multi_modal_projector|mm_projector/.test(t.name));
    const bytesOf = ts => ts.reduce((a, t) => a + t.n_bytes, 0);

    // Every block is the same shape, so one line describing a block says more
    // than twelve identical boxes would.
    const perBlock = BLOCK_ROLES
      .map(role => ({{ role: role, bytes: bytesOf(visTensors.filter(t => classifyBlockTensor(t.name) === role.key)) }}))
      .filter(g => g.bytes > 0)
      .sort((a, b) => a.role.order - b.role.order)
      .map(g => `<span style="color:rgb(${{g.role.rgb}});">${{g.role.label}}</span> ${{fmtBytes(g.bytes / numVision)}}`)
      .join(" &middot; ");

    const stage = (title, note, ts) => ts.length === 0 ? "" : `
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;padding:8px 0;border-top:1px solid var(--border);">
        <span style="font-size:13px;">${{title}}<span style="color:var(--text2);"> &mdash; ${{note}}</span></span>
        <span style="font-size:12px;color:var(--text2);font-family:var(--mono);white-space:nowrap;">${{fmtBytes(bytesOf(ts))}}</span>
      </div>`;

    html += `<div style="border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:20px;background:var(--bg2);">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px;">
        <span style="font-weight:600;">Vision encoder
          <span class="info-btn" onclick="showInfo(this,'Vision Encoder','This model carries a <strong>vision tower</strong> (tensors named <code>v.blk.N.*</code>) that encodes images into the same representation space the language model uses. It is a separate stack: images go through it, text does not. The image is first cut into fixed-size <strong>patches</strong>, each patch is embedded much like a token, and a <strong>position embedding</strong> records where in the picture it came from. The blocks then run over those patches exactly as a text transformer runs over tokens. A <strong>projector</strong> at the end maps the result into the language model&#39;s dimension, which is what lets the two stacks meet.')">i</span>
        </span>
        <span style="font-size:12px;color:var(--text2);font-family:var(--mono);">${{numVision}} blocks &middot; ${{visTensors.length}} tensors &middot; ${{fmtBytes(visBytes)}}</span>
      </div>
      ${{stage("Patch embedding", "cuts the image into patches", patch)}}
      ${{stage("Position embedding", "records where each patch came from", posEmbed)}}
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;padding:8px 0;border-top:1px solid var(--border);">
        <span style="font-size:13px;">${{numVision}} vision blocks<span style="color:var(--text2);"> &mdash; each ${{perBlock}}</span></span>
        <span style="font-size:12px;color:var(--text2);font-family:var(--mono);white-space:nowrap;">${{fmtBytes(visBytes)}}</span>
      </div>
      ${{stage("Output normalization", "stabilizes the tower's output before it is projected", DATA.tensors.filter(t => /^v\\..*(post_ln|post_layernorm)/.test(t.name)))}}
      ${{stage("Projector", "maps the tower's output into the language model's dimension", projector)}}
    </div>`;
  }}

  if (numLayers === 0) {{
    html += `<p style="color:var(--text2);font-size:13px;">
      This file has no text transformer blocks &mdash; it holds only the components shown above.
      Projector and vision-only files (such as <code>mmproj-*.gguf</code>) are packaged separately from the language model.
    </p></div>`;
    document.getElementById("archDiagram").innerHTML = html;
    return;
  }}

  // Embedding
  const embSize = embeds.length > 0 ? fmtBytes(embeds.reduce((a,t) => a+t.n_bytes, 0)) : "";
  html += `<div style="background:var(--accent2);color:#fff;padding:14px 20px;border-radius:8px;text-align:center;margin-bottom:2px;font-weight:600;">
    Token Embedding <span style="font-weight:400;opacity:0.8;font-size:12px;">${{embSize}}</span>
    <span class="info-btn" style="border-color:rgba(255,255,255,0.3);color:rgba(255,255,255,0.7);" onclick="event.stopPropagation();showInfo(this,'Token Embedding','${{TENSOR_EXPLANATIONS.token_embd.replace(/'/g,"&#39;")}}')">i</span>
  </div>`;
  html += `<div style="text-align:center;color:var(--text2);font-size:18px;line-height:1;">&#8595;</div>`;

  // Transformer blocks
  const showDetailed = numLayers <= 6;
  const layerKeys = Object.keys(layers).map(Number).sort((a,b) => a-b);

  // What roles a block is made of. Two blocks with the same signature really
  // are interchangeable in the diagram; two with different signatures are not.
  const signature = idx => [...new Set(
    layers[idx].map(t => classifyBlockTensor(t.name)).filter(k => k !== "norm" && k !== "other")
  )].sort().join("+");

  // Consecutive blocks that share a signature, as runs.
  const runs = [];
  layerKeys.forEach(idx => {{
    const sig = signature(idx);
    const last = runs[runs.length - 1];
    if (last && last.sig === sig) last.keys.push(idx);
    else runs.push({{ sig: sig, keys: [idx] }});
  }});

  const collapsed = (n, note) => `<div style="border:1px dashed var(--border);border-radius:8px;padding:16px;text-align:center;color:var(--text2);margin-bottom:2px;font-size:13px;">
      &#8942; ${{n}} more ${{note}} &#8942;
    </div><div style="text-align:center;color:var(--text2);font-size:18px;line-height:1;">&#8595;</div>`;

  if (showDetailed) {{
    layerKeys.forEach(idx => {{
      html += buildLayerBlock(idx, layers[idx], moe);
    }});
  }} else if (runs.length === 1) {{
    // Every block is the same: show the first two and the last two.
    layerKeys.slice(0, 2).forEach(idx => {{
      html += buildLayerBlock(idx, layers[idx], moe);
    }});
    html += collapsed(numLayers - 4, `${{blockNoun}}s (same structure)`);
    layerKeys.slice(-2).forEach(idx => {{
      html += buildLayerBlock(idx, layers[idx], moe);
    }});
  }} else {{
    // A hybrid stack. Collapsing the middle here would hide the blocks that
    // differ -- in Granite 4.0-H the four attention blocks sit at 5, 15, 25
    // and 35, entirely inside what a fixed first-two/last-two window drops,
    // and the summary would have claimed "same structure" about them.
    // Each run of identical blocks is collapsed on its own instead, so every
    // structural change in the stack stays visible.
    runs.forEach(run => {{
      html += buildLayerBlock(run.keys[0], layers[run.keys[0]], moe);
      if (run.keys.length === 2) {{
        html += buildLayerBlock(run.keys[1], layers[run.keys[1]], moe);
      }} else if (run.keys.length > 2) {{
        const last = run.keys[run.keys.length - 1];
        html += collapsed(run.keys.length - 1, `identical blocks, through block ${{last}}`);
      }}
    }});
  }}

  // Output norm
  if (norms.length > 0) {{
    html += `<div style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:12px 20px;text-align:center;margin-bottom:2px;color:var(--text2);font-size:13px;">
      Output Normalization <span style="font-size:11px;">(${{fmtBytes(norms[0].n_bytes)}})</span>
    </div>`;
    html += `<div style="text-align:center;color:var(--text2);font-size:18px;line-height:1;">&#8595;</div>`;
  }}

  // Output projection
  if (outputs.length > 0) {{
    html += `<div style="background:var(--accent2);color:#fff;padding:14px 20px;border-radius:8px;text-align:center;font-weight:600;">
      Output Projection <span style="font-weight:400;opacity:0.8;font-size:12px;">${{fmtBytes(outputs[0].n_bytes)}}</span>
      <span class="info-btn" style="border-color:rgba(255,255,255,0.3);color:rgba(255,255,255,0.7);" onclick="event.stopPropagation();showInfo(this,'Output Projection','${{TENSOR_EXPLANATIONS.output.replace(/'/g,"&#39;")}}')">i</span>
    </div>`;
  }}

  html += `</div>`;
  document.getElementById("archDiagram").innerHTML = html;
}}

function buildLayerBlock(idx, tensors, moe) {{
  const totalBytes = tensors.reduce((a,t) => a+t.n_bytes, 0);

  // Only draw boxes for the roles this block actually contains. A dense
  // transformer gets two; a Mamba block gets one; a hybrid model gets whatever
  // that particular block holds, which is the point.
  const groups = BLOCK_ROLES
    .map(role => ({{ role: role, tensors: tensors.filter(t => classifyBlockTensor(t.name) === role.key) }}))
    .filter(g => g.tensors.length > 0)
    // Array order is match order (`moe` has to be tried before `ffn`, whose
    // names it shares); `order` is draw order, so a block always reads in the
    // sequence the data flows through it: mixing first, then the per-token
    // transform.
    .sort((a, b) => a.role.order - b.role.order);

  let boxes = "";
  if (groups.length === 0) {{
    // An architecture we do not recognise. Saying nothing beats drawing empty
    // boxes labelled with parts the model does not have.
    const kinds = [...new Set(tensors.map(
      t => t.name.replace(/^(?:v\\.)?blk\\.\d+\\.|^(?:model\\.)?layers\\.\d+\\./, "")))].slice(0, 6);
    boxes = `<div style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:10px;text-align:center;">
      <div style="font-size:12px;font-weight:600;">Block weights</div>
      <div style="font-size:11px;color:var(--text2);margin-top:2px;font-family:var(--mono);">${{kinds.join(", ")}}</div>
      <div style="font-size:11px;color:var(--text2);">${{fmtBytes(totalBytes)}}</div>
    </div>`;
  }} else {{
    boxes = groups.map(g => {{
      const bytes = g.tensors.reduce((a,t) => a+t.n_bytes, 0);
      let parts = g.role.parts;
      if (g.role.key === "moe" && moe && moe.total) {{
        parts = moe.used
          ? `${{moe.total}} experts, ${{moe.used}} active per token`
          : `${{moe.total}} experts`;
      }}
      return `<div style="background:rgba(${{g.role.rgb}},0.1);border:1px solid rgba(${{g.role.rgb}},0.25);border-radius:6px;padding:10px;text-align:center;">
        <div style="font-size:12px;font-weight:600;color:rgb(${{g.role.rgb}});">${{g.role.label}}</div>
        <div style="font-size:11px;color:var(--text2);margin-top:2px;">${{parts}}</div>
        <div style="font-size:11px;color:var(--text2);">${{fmtBytes(bytes)}}</div>
      </div>`;
    }}).join("");
  }}

  let html = `<div style="border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:2px;background:var(--bg2);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <span style="font-weight:600;font-size:14px;">Block ${{idx}}</span>
      <span style="font-size:12px;color:var(--text2);font-family:var(--mono);">${{tensors.length}} tensors &middot; ${{fmtBytes(totalBytes)}}</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;">
      ${{boxes}}
    </div>
  </div>`;
  html += `<div style="text-align:center;color:var(--text2);font-size:18px;line-height:1;">&#8595;</div>`;
  return html;
}}

buildArchDiagram();

// Stat info for detail items
const STAT_EXPLANATIONS = {{
  "Mean": "The <strong>average</strong> of all weight values. If close to 0, the weights are balanced between positive and negative. A large drift from 0 might indicate the model has strong biases.",
  "Std": "The <strong>standard deviation</strong> measures how spread out the weight values are. Higher = more variation between weights. If extremely large or near 0, training may have had issues.",
  "Min": "The <strong>smallest</strong> weight value in this tensor. Very large negative values might indicate training instability or outliers.",
  "Max": "The <strong>largest</strong> weight value in this tensor. Very large positive values might indicate training instability or outliers.",
  "Abs Mean": "The <strong>average absolute value</strong> of all weights. This tells you the typical magnitude regardless of sign. Useful for comparing how 'active' different layers are.",
  "Zeros": "The <strong>percentage of weights that are exactly zero</strong>. High zeros in a non-quantized model might indicate pruning (intentional removal of unimportant weights) or dead neurons.",
}};

// Tensor list
function renderTensors(filter, dtype) {{
  const list = document.getElementById("tensorList");
  const fl = filter.toLowerCase();
  let count = 0;
  let html = "";

  DATA.tensors.forEach((t, idx) => {{
    if (fl && !t.name.toLowerCase().includes(fl)) return;
    if (dtype && t.dtype !== dtype) return;
    count++;

    const shapeStr = "[" + t.shape.join(", ") + "]";
    const s = t.stats;
    const explanation = getTensorExplanation(t.name);
    let detailHTML = `<div style="margin-top:12px;margin-bottom:8px;display:flex;align-items:flex-start;gap:8px;">
      <span class="info-btn" onclick="event.stopPropagation();showInfo(this,'${{t.name}}','${{explanation.replace(/'/g,"&#39;")}}')">i</span>
      <span style="font-size:12px;color:var(--text2);line-height:1.5;">Click the <em>i</em> button to learn what this tensor does in the model.</span>
    </div>`;

    if (s && !s.quantized && !s.error) {{
      const items = ["Mean","Std","Min","Max","Abs Mean","Zeros"];
      const vals = [s.mean,s.std,s.min,s.max,s.abs_mean,s.zeros_pct];
      detailHTML += `<div class="detail-grid">`;
      items.forEach((label, i) => {{
        const statInfo = STAT_EXPLANATIONS[label] || "";
        const displayVal = label === "Zeros" ? vals[i].toFixed(1) + "%" : fmtSci(vals[i]);
        detailHTML += `<div class="detail-item">
          <div class="dlabel">${{label}} <span class="info-btn" onclick="event.stopPropagation();showInfo(this,'${{label}}','${{statInfo.replace(/'/g,"&#39;")}}')">i</span></div>
          <div class="dvalue">${{displayVal}}</div>
        </div>`;
      }});
      detailHTML += `</div>`;
    }} else if (s && s.quantized) {{
      detailHTML += `<div class="detail-grid">
        <div class="detail-item"><div class="dlabel">Type</div><div class="dvalue">${{t.dtype}}</div></div>
        <div class="detail-item"><div class="dlabel">Raw Size</div><div class="dvalue">${{fmtBytes(s.n_bytes)}}</div></div>
      </div><p style="color:var(--text2);font-size:12px;margin-top:8px;">Detailed stats are not available for quantized tensors &mdash; the raw values are compressed and cannot be directly interpreted as individual numbers.</p>`;
    }}

    if (t.heatmap) {{
      detailHTML += `<div class="heatmap-wrap">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
          <span style="font-size:12px;color:var(--text2);">Weight heatmap</span>
          <span class="info-btn" onclick="event.stopPropagation();showInfo(this,'Weight Heatmap','This is a <strong>sampled visualization</strong> of the tensor values. Each pixel represents one weight. <strong>Blue = negative</strong> values, <strong>yellow/orange = positive</strong> values, with intensity showing magnitude. Patterns here can reveal structure: attention matrices often show diagonal patterns, while random-looking noise is normal for well-trained dense layers.')">i</span>
        </div>
        <canvas id="hm-${{idx}}" width="${{t.heatmap[0].length}}" height="${{t.heatmap.length}}"></canvas>
      </div>`;
    }}

    const shapeInfo = t.shape.length === 1
      ? "This is a <strong>1D tensor</strong> (a vector) with " + t.shape[0] + " values. Typically used for bias terms or layer normalization parameters."
      : "This is a <strong>2D tensor</strong> (a matrix) with " + t.shape[0] + " rows and " + t.shape[1] + " columns = " + t.n_elements.toLocaleString() + " values. Matrix multiplication with this tensor transforms input vectors from one representation to another.";

    html += `<div class="tensor" data-idx="${{idx}}">
      <div class="tensor-header" onclick="toggleTensor(this)">
        <span class="tensor-name">${{t.name}}</span>
        <span class="tensor-meta">${{shapeStr}} <span class="info-btn" onclick="event.stopPropagation();showInfo(this,'Shape: ${{shapeStr}}','${{shapeInfo.replace(/'/g,"&#39;")}}')">i</span></span>
        <span class="tensor-badge" style="border-color:${{dtypeColor(t.dtype)}};color:${{dtypeColor(t.dtype)}}">${{t.dtype}}</span>
        <span class="tensor-meta">${{fmtBytes(t.n_bytes)}}</span>
      </div>
      <div class="tensor-detail">${{detailHTML}}</div>
    </div>`;
  }});

  list.innerHTML = html;
  document.getElementById("tensorCount").textContent = count + " / " + DATA.tensors.length + " tensors";
}}

function toggleTensor(header) {{
  const tensor = header.parentElement;
  const wasOpen = tensor.classList.contains("open");
  tensor.classList.toggle("open");

  if (!wasOpen) {{
    const idx = parseInt(tensor.dataset.idx);
    const t = DATA.tensors[idx];
    if (t.heatmap) {{
      const canvas = document.getElementById("hm-" + idx);
      if (canvas && !canvas.dataset.drawn) {{
        drawHeatmap(canvas, t.heatmap);
        canvas.dataset.drawn = "1";
      }}
    }}
  }}
}}

function drawHeatmap(canvas, data) {{
  const ctx = canvas.getContext("2d");
  const h = data.length, w = data[0].length;
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {{
    for (let x = 0; x < w; x++) {{
      const v = data[y][x];
      const i = (y * w + x) * 4;
      if (v >= 0) {{
        img.data[i] = Math.round(v * 100);
        img.data[i+1] = Math.round(100 + v * 155);
        img.data[i+2] = Math.round(255);
      }} else {{
        img.data[i] = Math.round(255);
        img.data[i+1] = Math.round(100 + (1+v) * 155);
        img.data[i+2] = Math.round((1+v) * 100);
      }}
      img.data[i+3] = 255;
    }}
  }}
  ctx.putImageData(img, 0, 0);
}}

// Metadata table
const metaBody = document.getElementById("metaBody");
Object.entries(DATA.meta).sort().forEach(([key, value]) => {{
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${{key}}</td><td>${{value}}</td>`;
  metaBody.appendChild(tr);
}});

// Search & filter
const searchInput = document.getElementById("search");
searchInput.addEventListener("input", () => renderTensors(searchInput.value, dtypeFilter.value));
dtypeFilter.addEventListener("change", () => renderTensors(searchInput.value, dtypeFilter.value));

renderTensors("", "");
</script>
</body>
</html>"""
