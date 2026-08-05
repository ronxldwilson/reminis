"""Command-line interface for reminis."""

import argparse
import sys
from pathlib import Path

from reminis import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="reminis",
        description="Store LLM weights in a SQLite database. Convert GGUF and "
                    "safetensors losslessly, query, diff, and package fine-tunes "
                    "as delta packs.",
    )
    parser.add_argument("--version", action="version", version=f"reminis {__version__}")

    sub = parser.add_subparsers(dest="command")

    # convert: GGUF or safetensors -> SQLite
    p_convert = sub.add_parser(
        "convert", help="Convert a GGUF or safetensors model to a SQLite database"
    )
    p_convert.add_argument(
        "input",
        help="A .gguf file, a .safetensors file, a model.safetensors.index.json, "
             "or a directory holding a safetensors model",
    )
    p_convert.add_argument("-o", "--output", help="Output database path (default: same name with .db)")
    p_convert.add_argument(
        "--format", choices=("auto", "gguf", "safetensors"), default="auto",
        help="Input format (default: detected from the path)",
    )
    p_convert.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # export: SQLite -> GGUF or safetensors
    p_export = sub.add_parser("export", help="Convert a SQLite database back to a model file")
    p_export.add_argument("input", help="Path to the SQLite database")
    p_export.add_argument("-o", "--output", help="Output path (default: same name, format's extension)")
    p_export.add_argument(
        "--format", choices=("auto", "gguf", "safetensors"), default="auto",
        help="Output format (default: whichever format the model was imported from)",
    )
    p_export.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # lora: peft adapter -> delta pack
    p_lora = sub.add_parser(
        "lora", help="Convert a peft LoRA adapter into a delta pack against a base model"
    )
    p_lora.add_argument("adapter", help="adapter_model.safetensors, or the directory holding it")
    p_lora.add_argument("base", help="Path to the base model database the adapter was trained on")
    p_lora.add_argument("-o", "--output", required=True, help="Output delta pack path")
    p_lora.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # info: show database summary
    p_info = sub.add_parser("info", help="Show summary info about a reminis database")
    p_info.add_argument("input", help="Path to the SQLite database")

    # view: generate and open interactive HTML viewer
    p_view = sub.add_parser("view", help="Open an interactive viewer for a reminis database")
    p_view.add_argument("input", help="Path to the SQLite database")
    p_view.add_argument("-o", "--output", help="Output HTML path (default: same name with .html)")
    p_view.add_argument("--no-open", action="store_true", help="Don't open in browser")

    # diff: compare two model databases
    p_diff = sub.add_parser("diff", help="Compare two model databases tensor by tensor")
    p_diff.add_argument("base", help="Path to the base model database")
    p_diff.add_argument("target", help="Path to the target model database")
    p_diff.add_argument("-o", "--output", help="Write a delta pack that turns base into target")
    p_diff.add_argument(
        "--lossy",
        nargs="?",
        const=0.01,
        type=float,
        metavar="TOL",
        help="Allow low-rank encoding with at most TOL relative error per tensor "
             "(default 0.01 = 1%%). Much smaller packs when the delta is genuinely "
             "low-rank, as a LoRA fine-tune's is. Off by default.",
    )
    p_diff.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # apply: apply a delta pack to a base model
    p_apply = sub.add_parser("apply", help="Apply a delta pack to a base model")
    p_apply.add_argument("base", help="Path to the base model database")
    p_apply.add_argument("delta", help="Path to the delta pack")
    p_apply.add_argument("-o", "--output", required=True, help="Output database path")
    p_apply.add_argument("--no-verify", action="store_true", help="Skip hash verification")
    p_apply.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "convert":
        fmt = args.format if args.format != "auto" else _detect_input_format(args.input)
        if fmt == "safetensors":
            from reminis.safetensors_io import safetensors_to_sqlite
            safetensors_to_sqlite(args.input, args.output, verbose=not args.quiet)
        else:
            from reminis.converter import gguf_to_sqlite
            gguf_to_sqlite(args.input, args.output, verbose=not args.quiet)

    elif args.command == "export":
        fmt = args.format if args.format != "auto" else _source_format(args.input)
        if fmt == "safetensors":
            from reminis.safetensors_io import sqlite_to_safetensors
            sqlite_to_safetensors(args.input, args.output, verbose=not args.quiet)
        else:
            from reminis.converter import sqlite_to_gguf
            sqlite_to_gguf(args.input, args.output, verbose=not args.quiet)

    elif args.command == "lora":
        from reminis.lora import lora_to_delta_pack
        lora_to_delta_pack(
            args.adapter, args.base, args.output, verbose=not args.quiet
        )

    elif args.command == "info":
        _show_info(args.input)

    elif args.command == "view":
        import webbrowser
        from reminis.viewer import generate_viewer
        html_path = generate_viewer(args.input, args.output)
        if not args.no_open:
            webbrowser.open("file://" + str(Path(html_path).resolve()))

    elif args.command == "diff":
        from reminis.diff import diff_models
        if args.lossy is not None and not args.output:
            parser.error("--lossy only affects the written pack; pass -o/--output too")
        diff_models(
            args.base, args.target, args.output,
            verbose=not args.quiet, lossy_tolerance=args.lossy,
        )

    elif args.command == "apply":
        from reminis.diff import apply_delta
        apply_delta(
            args.base, args.delta, args.output,
            verify=not args.no_verify, verbose=not args.quiet,
        )


def _detect_input_format(path: str) -> str:
    """Guess the input format from the path.

    A directory or an index file only ever means safetensors; GGUF is always a
    single file. Anything else falls through to GGUF, which is what reminis
    read before safetensors support existed.
    """
    p = Path(path)
    if p.is_dir() or p.name.endswith(".index.json") or p.suffix == ".safetensors":
        return "safetensors"
    return "gguf"


def _source_format(db_path: str) -> str:
    """The format a database was imported from, so export defaults to it."""
    import sqlite3

    if not Path(db_path).exists():
        return "gguf"
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM model_meta WHERE key = 'reminis.source_format'"
        ).fetchone()
    except sqlite3.DatabaseError:
        return "gguf"
    finally:
        conn.close()
    return row[0] if row else "gguf"


def _show_info(db_path: str):
    import sqlite3
    from pathlib import Path

    path = Path(db_path)
    if not path.exists():
        print(f"Error: {db_path} not found")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Database: {db_path} ({size_mb:.1f} MB)")

    # Model name, architecture, and which format it came from
    for key in ("general.name", "general.architecture", "reminis.source_format"):
        row = conn.execute("SELECT value FROM model_meta WHERE key = ?", (key,)).fetchone()
        if row:
            print(f"  {key}: {row[0]}")

    # Counts
    meta_count = conn.execute("SELECT COUNT(*) FROM model_meta").fetchone()[0]
    tensor_count = conn.execute("SELECT COUNT(*) FROM tensors").fetchone()[0]
    total_elements = conn.execute("SELECT SUM(n_elements) FROM tensors").fetchone()[0] or 0
    total_bytes = conn.execute("SELECT SUM(n_bytes) FROM tensors").fetchone()[0] or 0

    print(f"  Metadata fields: {meta_count}")
    print(f"  Tensors: {tensor_count}")
    print(f"  Parameters: {total_elements:,}")
    print(f"  Weight data: {total_bytes / (1024 * 1024):.1f} MB")

    # Dtype breakdown
    print(f"\n  Dtype breakdown:")
    for dtype, count, total in conn.execute(
        "SELECT dtype, COUNT(*), SUM(n_bytes) FROM tensors GROUP BY dtype ORDER BY SUM(n_bytes) DESC"
    ):
        print(f"    {dtype:10s}  {count:4d} tensors  {total / (1024 * 1024):8.1f} MB")

    conn.close()
