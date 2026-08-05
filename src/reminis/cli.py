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

    # log: inspect a training log
    p_log = sub.add_parser("log", help="Inspect a training log written by reminis.track")
    p_log.add_argument("input", help="Path to the training log database")
    p_log.add_argument("--step", type=int, help="Show per-parameter detail for one step")
    p_log.add_argument("--spikes", action="store_true", help="Show only loss spikes")

    # rollback: restore a model to a snapshot
    p_rollback = sub.add_parser(
        "rollback", help="Restore a model to its weights at a logged snapshot"
    )
    p_rollback.add_argument("log", help="Path to the training log database")
    p_rollback.add_argument("step", type=int, help="Snapshot step to restore")
    p_rollback.add_argument("-o", "--output", required=True, help="Output database path")

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

    elif args.command == "log":
        _show_log(args.input, step=args.step, spikes_only=args.spikes)

    elif args.command == "rollback":
        from reminis.track import TrainingLog, rollback_to_step
        log = TrainingLog(args.log)
        try:
            rollback_to_step(log, args.step, args.output, verbose=True)
        finally:
            log.close()

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


def _show_log(log_path: str, step: int | None = None, spikes_only: bool = False):
    """Print what a training run did, straight out of SQL."""
    from reminis.track import TrainingLog

    if not Path(log_path).exists():
        print(f"Error: {log_path} not found")
        sys.exit(1)

    log = TrainingLog(log_path)
    try:
        meta = dict(log.conn.execute("SELECT key, value FROM run_meta"))
        n_steps = log.conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        n_updates = log.conn.execute("SELECT COUNT(*) FROM param_updates").fetchone()[0]
        n_snaps = log.conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]

        size_mb = Path(log_path).stat().st_size / (1024 * 1024)
        print(f"Training log: {log_path} ({size_mb:.2f} MB)")
        print(f"  Run: {meta.get('run_name', 'unknown')}")
        for key in ("model_class", "learning_rate", "num_train_epochs",
                    "logging_overhead_seconds"):
            if key in meta:
                print(f"  {key}: {meta[key]}")
        print(f"  Steps logged: {n_steps}")
        print(f"  Parameter updates: {n_updates:,}")
        print(f"  Snapshots: {n_snaps}")

        if step is not None:
            print(f"\n  Step {step}, largest gradients:")
            rows = log.step_detail(step)
            if not rows:
                print("    (no rows for that step)")
            for param, gnorm, gmax, wnorm in rows:
                print(f"    {param:<50s} |grad|={gnorm:>10.4f}  "
                      f"max={gmax:>9.4f}  |w|={wnorm if wnorm else 0:>10.4f}")
            return

        spikes = log.loss_spikes()
        if spikes:
            print(f"\n  Loss spikes ({len(spikes)}):")
            for spike_step, before, after in spikes[:10]:
                print(f"    step {spike_step:>5}: {before:.4f} -> {after:.4f} "
                      f"({after / before:.2f}x)")
            print("    Inspect one with: reminis log <log.db> --step <N>")
        elif spikes_only:
            print("\n  No loss spikes detected.")

        if spikes_only:
            return

        curve = log.loss_curve()
        if curve:
            first, last = curve[0], curve[-1]
            best = min(curve, key=lambda r: r[1])
            print(f"\n  Loss: {first[1]:.4f} (step {first[0]}) -> "
                  f"{last[1]:.4f} (step {last[0]}), best {best[1]:.4f} at step {best[0]}")

        top = log.most_changed_params(limit=10)
        if top:
            print(f"\n  Most-updated parameters (by cumulative gradient norm):")
            for param, total, n in top:
                print(f"    {param:<50s} {total:>12.2f}  over {n} steps")

        snaps = log.conn.execute(
            "SELECT step, kind, base_step, bytes FROM snapshots ORDER BY step"
        ).fetchall()
        if snaps:
            total = sum(s[3] for s in snaps)
            print(f"\n  Snapshots ({_fmt(total)} total):")
            for snap_step, kind, base_step, size in snaps:
                against = f" vs step {base_step}" if base_step is not None else ""
                print(f"    step {snap_step:>5}  {kind:<6s}{against:<15s} {_fmt(size)}")
            print(f"\n  Restore one with: reminis rollback <log.db> <step> -o restored.db")
    finally:
        log.close()


def _fmt(b: int) -> str:
    for unit, size in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if b >= size:
            return f"{b / size:.1f} {unit}"
    return f"{b} B"


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
