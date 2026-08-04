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

<div class="stats" id="statsGrid"></div>

<div id="dtypeViz"></div>

<div class="tabs">
  <button class="tab active" data-tab="tensors">Tensors</button>
  <button class="tab" data-tab="metadata">Metadata</button>
</div>

<div class="tab-content active" id="tab-tensors">
  <div class="toolbar">
    <input type="text" id="search" placeholder="Search tensors...">
    <select id="dtypeFilter"><option value="">All types</option></select>
    <span class="count" id="tensorCount"></span>
  </div>
  <div id="tensorList"></div>
</div>

<div class="tab-content" id="tab-metadata">
  <table class="meta-table" id="metaTable">
    <thead><tr><th>Key</th><th>Value</th></tr></thead>
    <tbody id="metaBody"></tbody>
  </table>
</div>

<div class="footer">
  Generated by <a href="https://github.com/ronxldwilson/reminis">reminis</a>
</div>

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

// Header
document.getElementById("modelName").textContent = DATA.model_name;
document.getElementById("headerSub").textContent =
  DATA.architecture + " | " + DATA.db_file + " (" + DATA.db_size_mb + " MB)";

// Stats
const statsHTML = [
  ["Parameters", fmtNum(DATA.total_params), DATA.total_params.toLocaleString()],
  ["Tensors", DATA.tensors.length.toString(), ""],
  ["Weight Data", fmtBytes(DATA.total_bytes), ""],
  ["Architecture", DATA.architecture, ""],
  ["Database", DATA.db_size_mb + " MB", DATA.db_file],
].map(([label, value, detail]) =>
  `<div class="stat"><div class="label">${{label}}</div><div class="value">${{value}}</div>${{detail ? `<div class="detail">${{detail}}</div>` : ""}}</div>`
).join("");
document.getElementById("statsGrid").innerHTML = statsHTML;

// Dtype breakdown
const dtypes = Object.entries(DATA.dtype_counts).sort((a,b) => b[1].bytes - a[1].bytes);
let barHTML = '<div class="dtype-bar">';
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
    let detailHTML = "";

    if (s && !s.quantized && !s.error) {{
      detailHTML = `<div class="detail-grid">
        <div class="detail-item"><div class="dlabel">Mean</div><div class="dvalue">${{fmtSci(s.mean)}}</div></div>
        <div class="detail-item"><div class="dlabel">Std</div><div class="dvalue">${{fmtSci(s.std)}}</div></div>
        <div class="detail-item"><div class="dlabel">Min</div><div class="dvalue">${{fmtSci(s.min)}}</div></div>
        <div class="detail-item"><div class="dlabel">Max</div><div class="dvalue">${{fmtSci(s.max)}}</div></div>
        <div class="detail-item"><div class="dlabel">Abs Mean</div><div class="dvalue">${{fmtSci(s.abs_mean)}}</div></div>
        <div class="detail-item"><div class="dlabel">Zeros</div><div class="dvalue">${{s.zeros_pct.toFixed(1)}}%</div></div>
      </div>`;
    }} else if (s && s.quantized) {{
      detailHTML = `<div class="detail-grid">
        <div class="detail-item"><div class="dlabel">Type</div><div class="dvalue">${{t.dtype}}</div></div>
        <div class="detail-item"><div class="dlabel">Raw Size</div><div class="dvalue">${{fmtBytes(s.n_bytes)}}</div></div>
      </div><p style="color:var(--text2);font-size:12px;margin-top:8px;">Stats not available for quantized tensors</p>`;
    }}

    if (t.heatmap) {{
      detailHTML += `<div class="heatmap-wrap"><canvas id="hm-${{idx}}" width="${{t.heatmap[0].length}}" height="${{t.heatmap.length}}"></canvas></div>`;
    }}

    html += `<div class="tensor" data-idx="${{idx}}">
      <div class="tensor-header" onclick="toggleTensor(this)">
        <span class="tensor-name">${{t.name}}</span>
        <span class="tensor-meta">${{shapeStr}}</span>
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
