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

    # run: generate text from weights in the database
    p_run = sub.add_parser(
        "run",
        help="Generate text from a model, reading its weights out of the database",
        description="A pure-numpy forward pass over tensors selected from "
                    "SQLite. No torch, no llama.cpp, no config files -- "
                    "everything comes out of the database.",
    )
    p_run.add_argument("input", help="Path to the model database")
    p_run.add_argument("prompt", help="The prompt to continue")
    p_run.add_argument("-n", "--max-tokens", type=int, default=64,
                       help="How many tokens to generate (default 64)")
    p_run.add_argument("--temp", type=float, default=0.8,
                       help="Sampling temperature; 0 is greedy (default 0.8)")
    p_run.add_argument("--top-p", type=float, default=0.95,
                       help="Nucleus sampling cutoff; 1 disables it (default 0.95)")
    p_run.add_argument("--seed", type=int, help="Seed, so a run repeats exactly")
    p_run.add_argument("--chat", action="store_true",
                       help="Wrap the prompt in the model's chat template")
    p_run.add_argument("--stream", action="store_true",
                       help="Re-read every weight from SQLite instead of caching "
                            "it, so peak memory is one layer rather than the "
                            "whole model. Much slower.")
    p_run.add_argument(
        "--backend", choices=("auto", "numpy", "mlx", "cupy"), default="auto",
        help="Which array library to compute with. auto picks the fastest "
             "one this machine has: mlx on Apple silicon, cupy on NVIDIA, "
             "numpy otherwise. numpy is the reference implementation.",
    )
    p_run.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # merge: combine several models into one
    p_merge = sub.add_parser(
        "merge",
        help="Merge several model databases into one",
        description="Align the models' tensors with a SQL join and combine "
                    "them elementwise. Float weights only -- quantized "
                    "tensors cannot be averaged meaningfully.",
    )
    p_merge.add_argument(
        "inputs", nargs="+",
        help="Two or more model databases. With --base, one is allowed too: "
             "--scale then rescales that single fine-tune, and a negative "
             "scale subtracts it from the base.",
    )
    p_merge.add_argument("-o", "--output", required=True, help="Output database path")
    p_merge.add_argument(
        "--method", choices=("linear", "slerp", "task-arithmetic", "ties"),
        default="linear",
        help="linear: weighted average. slerp: spherical interpolation between "
             "two. task-arithmetic / ties: combine (model - base) task vectors, "
             "which need --base. Default: linear.",
    )
    p_merge.add_argument(
        "--weights",
        help="Comma-separated weight per input (default: equal). Linear "
             "weights are normalised to sum to 1.",
    )
    p_merge.add_argument(
        "--base",
        help="For task-arithmetic and ties: the checkpoint the inputs were "
             "fine-tuned from",
    )
    p_merge.add_argument(
        "-t", type=float, default=0.5, metavar="T",
        help="For slerp: how far from the first model to the second (default 0.5)",
    )
    p_merge.add_argument(
        "--density", type=float, default=0.2,
        help="For ties: fraction of each task vector kept when trimming (default 0.2)",
    )
    p_merge.add_argument(
        "--scale", type=float, default=1.0,
        help="Multiplier on the combined task vector before it is added back "
             "to the base (default 1.0)",
    )
    p_merge.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # registry: many models in one database
    p_reg = sub.add_parser(
        "registry",
        help="Keep many related models in one database, derived ones as deltas",
    )
    reg_sub = p_reg.add_subparsers(dest="registry_command")

    r_add = reg_sub.add_parser("add", help="Add a model to a registry")
    r_add.add_argument("registry", help="Path to the registry database (created if new)")
    r_add.add_argument("source", help="A .gguf, .safetensors, model directory, or .db")
    r_add.add_argument("--name", required=True, help="Name to store it under")
    r_add.add_argument(
        "--parent",
        help="Store only the difference from this already-registered model. "
             "Without it, the model is stored in full as a new base.",
    )
    r_add.add_argument(
        "--lossy", nargs="?", const=0.01, type=float, metavar="TOL",
        help="With --parent, allow low-rank encoding within TOL relative error",
    )
    r_add.add_argument("--notes", help="Free-text note stored alongside the model")
    r_add.add_argument("-q", "--quiet", action="store_true")

    r_lora = reg_sub.add_parser(
        "add-lora", help="Add a peft LoRA adapter as a derived model"
    )
    r_lora.add_argument("registry", help="Path to the registry database")
    r_lora.add_argument("adapter", help="adapter_model.safetensors, or its directory")
    r_lora.add_argument("--name", required=True, help="Name to store it under")
    r_lora.add_argument("--parent", required=True, help="The base it was trained on")
    r_lora.add_argument("--notes", help="Free-text note stored alongside the model")
    r_lora.add_argument("-q", "--quiet", action="store_true")

    r_ls = reg_sub.add_parser("ls", help="List the models in a registry")
    r_ls.add_argument("registry", help="Path to the registry database")

    r_export = reg_sub.add_parser("export", help="Get a model back out of a registry")
    r_export.add_argument("registry", help="Path to the registry database")
    r_export.add_argument("--name", required=True, help="Model to export")
    r_export.add_argument(
        "-o", "--output", required=True,
        help="Output path. A .db writes a single-model database; .gguf or "
             ".safetensors writes that format.",
    )
    r_export.add_argument("-q", "--quiet", action="store_true")

    r_rm = reg_sub.add_parser("rm", help="Remove a model from a registry")
    r_rm.add_argument("registry", help="Path to the registry database")
    r_rm.add_argument("--name", required=True, help="Model to remove")

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
        _reject_registry(args.input, "info")
        _show_info(args.input)

    elif args.command == "view":
        _reject_registry(args.input, "view")
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

    elif args.command == "run":
        _reject_registry(args.input, "run")
        from reminis.infer import run_cli
        run_cli(args)

    elif args.command == "merge":
        from reminis.merge import merge_models
        for path in args.inputs + ([args.base] if args.base else []):
            _reject_registry(path, "merge")
        weights = None
        if args.weights:
            try:
                weights = [float(w) for w in args.weights.split(",")]
            except ValueError:
                parser.error("--weights must be comma-separated numbers, e.g. 0.7,0.3")
        try:
            merge_models(
                args.inputs, args.output, method=args.method, weights=weights,
                base=args.base, density=args.density, t=args.t, scale=args.scale,
                verbose=not args.quiet,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    elif args.command == "registry":
        if args.registry_command is None:
            p_reg.print_help()
            sys.exit(1)
        _run_registry(args, p_reg)

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


def _is_registry(db_path: str) -> bool:
    """True if this database holds many models rather than one."""
    import sqlite3

    if not Path(db_path).exists():
        return False
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='models'"
        ).fetchone() is not None
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()


def _reject_registry(db_path: str, command: str):
    """Single-model commands must not silently aggregate a whole registry.

    `SELECT COUNT(*) FROM tensors` on a registry returns every model's tensors
    added together, which looks like an answer and is not one.
    """
    if _is_registry(db_path):
        print(
            f"Error: {db_path} is a registry holding several models, and "
            f"`reminis {command}` works on one model.\n"
            f"  List what is inside:   reminis registry ls {db_path}\n"
            f"  Get one model out:     reminis registry export {db_path} "
            f"--name <name> -o model.db"
        )
        sys.exit(1)


def _run_registry(args, parser):
    from reminis.registry import Registry

    command = args.registry_command
    # Only `add` may create a registry; the rest operating on a missing path is
    # a typo, and silently making an empty one would hide it.
    registry = Registry(args.registry, create=command in ("add", "add-lora"))

    try:
        if command == "add":
            if args.lossy is not None and not args.parent:
                parser.error("--lossy only applies with --parent")
            verbose = not args.quiet
            if args.parent:
                registry.add_derived(
                    args.source, args.name, args.parent,
                    lossy_tolerance=args.lossy, notes=args.notes, verbose=verbose,
                )
            else:
                registry.add_base(
                    args.source, args.name, notes=args.notes, verbose=verbose
                )
            if verbose:
                _print_registry_summary(registry)

        elif command == "add-lora":
            registry.add_lora(
                args.adapter, args.name, args.parent,
                notes=args.notes, verbose=not args.quiet,
            )
            if not args.quiet:
                _print_registry_summary(registry)

        elif command == "ls":
            _list_registry(registry)

        elif command == "export":
            out = Path(args.output)
            if out.suffix == ".db":
                registry.materialize(args.name, args.output, verbose=not args.quiet)
            else:
                registry.export(args.name, args.output, verbose=not args.quiet)

        elif command == "rm":
            registry.remove(args.name)
            _print_registry_summary(registry)
    finally:
        registry.close()


def _list_registry(registry):
    models = registry.list_models()
    if not models:
        print(f"{registry.path} has no models yet.")
        print("Add one with: reminis registry add <registry.db> <model> --name <name>")
        return

    print(f"Registry: {registry.path}\n")
    print(f"  {'NAME':<28} {'KIND':<8} {'PARENT':<20} {'TENSORS':>8} "
          f"{'FULL SIZE':>11} {'STORED':>11} {'':>7}")
    print("  " + "-" * 96)

    by_parent = {}
    for m in models:
        by_parent.setdefault(m["parent"], []).append(m)

    def emit(parent, depth):
        for m in by_parent.get(parent, []):
            indent = "  " * depth
            label = indent + m["name"]
            pct = (
                f"{m['stored_bytes'] / m['logical_bytes'] * 100:.1f}%"
                if m["logical_bytes"] else ""
            )
            flag = " lossy" if m["lossy"] else ""
            print(f"  {label:<28} {m['kind']:<8} {(m['parent'] or '-'):<20} "
                  f"{m['n_tensors']:>8} {_fmt(m['logical_bytes']):>11} "
                  f"{_fmt(m['stored_bytes']):>11} {pct:>7}{flag}")
            emit(m["name"], depth + 1)

    emit(None, 0)

    s = registry.stats()
    print("  " + "-" * 96)
    print(f"\n  {s['models']} models ({s['bases']} base, {s['derived']} derived)")
    print(f"  Stored separately, these would be: {_fmt(s['logical_bytes'])}")
    print(f"  This registry file is:             {_fmt(s['file_bytes'])}")
    if s["savings"] > 0:
        print(f"  Saved: {s['savings'] * 100:.1f}%  "
              f"({_fmt(s['logical_bytes'] - s['file_bytes'])})")


def _print_registry_summary(registry):
    s = registry.stats()
    print(f"\n  Registry now holds {s['models']} models, "
          f"{_fmt(s['file_bytes'])} on disk "
          f"(vs {_fmt(s['logical_bytes'])} stored separately)")


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

    from reminis.backend import report as backend_report
    print(f"\n  Compute backends:")
    print(backend_report())

    conn.close()
