"""Command-line interface for reminis."""

import argparse
import sys
from pathlib import Path

from reminis import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="reminis",
        description="Store LLM weights in a SQLite database. Query, diff, rollback, and merge.",
    )
    parser.add_argument("--version", action="version", version=f"reminis {__version__}")

    sub = parser.add_subparsers(dest="command")

    # convert: GGUF -> SQLite
    p_convert = sub.add_parser("convert", help="Convert a GGUF file to a SQLite database")
    p_convert.add_argument("input", help="Path to the GGUF file")
    p_convert.add_argument("-o", "--output", help="Output database path (default: same name with .db)")
    p_convert.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # export: SQLite -> GGUF
    p_export = sub.add_parser("export", help="Convert a SQLite database back to GGUF")
    p_export.add_argument("input", help="Path to the SQLite database")
    p_export.add_argument("-o", "--output", help="Output GGUF path (default: same name with .gguf)")
    p_export.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # info: show database summary
    p_info = sub.add_parser("info", help="Show summary info about a reminis database")
    p_info.add_argument("input", help="Path to the SQLite database")

    # view: generate and open interactive HTML viewer
    p_view = sub.add_parser("view", help="Open an interactive viewer for a reminis database")
    p_view.add_argument("input", help="Path to the SQLite database")
    p_view.add_argument("-o", "--output", help="Output HTML path (default: same name with .html)")
    p_view.add_argument("--no-open", action="store_true", help="Don't open in browser")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "convert":
        from reminis.converter import gguf_to_sqlite
        gguf_to_sqlite(args.input, args.output, verbose=not args.quiet)

    elif args.command == "export":
        from reminis.converter import sqlite_to_gguf
        sqlite_to_gguf(args.input, args.output, verbose=not args.quiet)

    elif args.command == "info":
        _show_info(args.input)

    elif args.command == "view":
        import webbrowser
        from reminis.viewer import generate_viewer
        html_path = generate_viewer(args.input, args.output)
        if not args.no_open:
            webbrowser.open("file://" + str(Path(html_path).resolve()))


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

    # Model name and architecture
    for key in ("general.name", "general.architecture"):
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
