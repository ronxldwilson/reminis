"""Command-line interface for reminis."""

import argparse
import sys
from pathlib import Path

from reminis import __version__


# Every command is a function taking (args, parser), attached to its
# subparser with `set_defaults(func=...)` so `main` dispatches by calling it.
# The alternative -- and what this was until 0.32.3 -- is a chain of
# `elif args.command == "..."` at the bottom of a 495-line function, three
# hundred lines from the flags each branch reads.
#
# `parser` is passed because a handler's remaining job is often to reject an
# argument combination argparse cannot express, and `parser.error` is what
# prints usage and exits 2 the way every other argparse failure does.


def cmd_convert(args, parser):
    fmt = args.format if args.format != "auto" else _detect_input_format(args.input)
    if fmt == "safetensors":
        from reminis.safetensors_io import safetensors_to_sqlite
        safetensors_to_sqlite(args.input, args.output, verbose=not args.quiet)
    else:
        from reminis.converter import gguf_to_sqlite
        gguf_to_sqlite(args.input, args.output, verbose=not args.quiet)


def cmd_export(args, parser):
    fmt = args.format if args.format != "auto" else _source_format(args.input)
    if fmt == "safetensors":
        from reminis.safetensors_io import sqlite_to_safetensors
        sqlite_to_safetensors(args.input, args.output, verbose=not args.quiet)
    else:
        from reminis.converter import sqlite_to_gguf
        sqlite_to_gguf(args.input, args.output, verbose=not args.quiet)


def cmd_lora(args, parser):
    from reminis.lora import lora_to_delta_pack
    lora_to_delta_pack(
        args.adapter, args.base, args.output, verbose=not args.quiet
    )


def cmd_info(args, parser):
    _reject_registry(args.input, "info")
    _show_info(args.input)


def cmd_view(args, parser):
    _reject_registry(args.input, "view")
    import webbrowser
    from reminis.viewer import generate_viewer
    html_path = generate_viewer(args.input, args.output)
    if not args.no_open:
        webbrowser.open("file://" + str(Path(html_path).resolve()))


def cmd_diff(args, parser):
    from reminis.diff import diff_models
    if args.lossy is not None and not args.output:
        parser.error("--lossy only affects the written pack; pass -o/--output too")
    diff_models(
        args.base, args.target, args.output,
        verbose=not args.quiet, lossy_tolerance=args.lossy,
    )


def cmd_run(args, parser):
    _reject_registry(args.input, "run")
    if args.pack is not None and args.pack != "native":
        if args.pack == "compact":
            pass
        elif args.pack in ("2", "3", "4", "5", "6", "8"):
            args.pack = int(args.pack)
        else:
            parser.error(
                "--pack takes no value (bit-exact), 'compact', or a width: "
                "2, 3, 4, 5, 6 or 8"
            )
    if args.experts is not None and args.experts != "all":
        if not args.experts.isdigit() or int(args.experts) < 1:
            parser.error("--experts takes a positive number, or 'all'")
        args.experts = int(args.experts)
    if args.draft_tokens < 1:
        parser.error("--draft-tokens must be at least 1")
    from reminis.infer import run_cli
    run_cli(args)


def cmd_transcribe(args, parser):
    import os

    _reject_registry(args.input, "transcribe")
    if not os.path.exists(args.audio):
        parser.error(f"no such audio file: {args.audio}")

    from reminis.backend import select as select_backend
    from reminis.whisper import UnsupportedModel, transcribe_file

    backend = None
    if args.backend != "auto":
        backend = select_backend("inference", args.backend)

    try:
        result = transcribe_file(
            args.input, args.audio, max_tokens=args.max_tokens,
            language=args.language, task=args.task, temperature=args.temp,
            seed=args.seed, backend=backend, verbose=not args.quiet,
        )
    except UnsupportedModel as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if not args.quiet:
        encoder, decoder = result["layers"]
        print(f"{encoder} encoder + {decoder} decoder layers | "
              f"{result['backend']}")
        rate = result["source_rate"]
        note = "" if rate == 16000 else f", resampled from {rate} Hz"
        print(f"{result['seconds']:.1f}s of audio{note}")
        if result["truncated"]:
            # Whisper's positional table is exactly 30 seconds long, so this
            # is a property of the model rather than a choice made here.
            print("Note: only the first 30s was transcribed -- Whisper's "
                  "encoder is exactly that long.")
        print()

    if result["text"] is None:
        print("This database carries no tokenizer, so only ids are available.")
        print(result["tokens"])
    else:
        print(result["text"].strip())

    if not args.quiet:
        print()
        print(f"{len(result['tokens'])} tokens in {result['elapsed']:.2f}s")
    if args.tokens:
        print(result["tokens"])


def cmd_merge(args, parser):
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


def cmd_registry(args, parser):
    # `registry` alone is not a command, so it prints its own help rather than
    # the top-level parser's. `registry_parser` is set by the same
    # `set_defaults` call that attached this handler.
    if args.registry_command is None:
        args.registry_parser.print_help()
        sys.exit(1)
    _run_registry(args, args.registry_parser)


def cmd_log(args, parser):
    _show_log(args.input, step=args.step, spikes_only=args.spikes)


def cmd_rollback(args, parser):
    from reminis.track import TrainingLog, rollback_to_step
    log = TrainingLog(args.log)
    try:
        rollback_to_step(log, args.step, args.output, verbose=True)
    finally:
        log.close()


def cmd_quantize(args, parser):
    from reminis.quantize import quantize_model
    quantize_model(args.database, args.output, bits=args.bits)


def cmd_sweep(args, parser):
    from reminis.sweep import mix_sweep, sweep

    widths = []
    for piece in args.bits.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if not piece.isdigit() or int(piece) not in (2, 3, 4, 5, 6, 8):
            parser.error(
                f"--bits takes a comma-separated list drawn from "
                f"2, 3, 4, 5, 6 and 8; got '{piece}'"
            )
        widths.append(int(piece))
    if not widths:
        parser.error("--bits needs at least one width")

    if args.mix and len(widths) < 2:
        parser.error("--mix needs at least two widths to choose between")
    if not 0.0 < args.budget <= 1.0:
        parser.error("--budget is a top-1 agreement fraction, between 0 and 1")

    run = mix_sweep if args.mix else sweep
    extra = {"budget": args.budget} if args.mix else {}
    run(
        args.database, widths, prompt=args.prompt,
        backend=None if args.backend == "auto" else args.backend,
        kv_bits=args.kv_bits, **extra,
    )


def cmd_blame(args, parser):
    _show_blame(args.log, args.param, args.top)


def cmd_bisect(args, parser):
    _run_bisect(args.log, args.good, args.bad, args.test)


def cmd_apply(args, parser):
    from reminis.diff import apply_delta
    apply_delta(
        args.base, args.delta, args.output,
        verify=not args.no_verify, verbose=not args.quiet,
    )


def build_parser() -> argparse.ArgumentParser:
    """The full command-line grammar.

    Separate from `main` so the parser can be built and inspected without
    running anything -- which is what the pre-release check does to assert
    every subcommand still reaches a handler.
    """
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
    p_convert.set_defaults(func=cmd_convert)

    # export: SQLite -> GGUF or safetensors
    p_export = sub.add_parser("export", help="Convert a SQLite database back to a model file")
    p_export.add_argument("input", help="Path to the SQLite database")
    p_export.add_argument("-o", "--output", help="Output path (default: same name, format's extension)")
    p_export.add_argument(
        "--format", choices=("auto", "gguf", "safetensors"), default="auto",
        help="Output format (default: whichever format the model was imported from)",
    )
    p_export.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    p_export.set_defaults(func=cmd_export)

    # lora: peft adapter -> delta pack
    p_lora = sub.add_parser(
        "lora", help="Convert a peft LoRA adapter into a delta pack against a base model"
    )
    p_lora.add_argument("adapter", help="adapter_model.safetensors, or the directory holding it")
    p_lora.add_argument("base", help="Path to the base model database the adapter was trained on")
    p_lora.add_argument("-o", "--output", required=True, help="Output delta pack path")
    p_lora.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    p_lora.set_defaults(func=cmd_lora)

    # info: show database summary
    p_info = sub.add_parser("info", help="Show summary info about a reminis database")
    p_info.add_argument("input", help="Path to the SQLite database")
    p_info.set_defaults(func=cmd_info)

    # view: generate and open interactive HTML viewer
    p_view = sub.add_parser("view", help="Open an interactive viewer for a reminis database")
    p_view.add_argument("input", help="Path to the SQLite database")
    p_view.add_argument("-o", "--output", help="Output HTML path (default: same name with .html)")
    p_view.add_argument("--no-open", action="store_true", help="Don't open in browser")
    p_view.set_defaults(func=cmd_view)

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
    p_diff.set_defaults(func=cmd_diff)

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
    p_run.add_argument(
        "--think", action="store_true",
        help="With --chat on a reasoning model, leave its thinking channel "
             "open so it works through the problem before answering. Closed "
             "by default: left open, a model reasons for as long as it likes "
             "and a short run shows working and no answer, which reads like "
             "a broken forward pass and is not one.",
    )
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
    p_run.add_argument(
        "--pack", nargs="?", const="native", metavar="BITS",
        help="Keep weights compressed in memory instead of expanding them to "
             "float16. Needs a backend that can multiply packed weights "
             "(mlx). "
             "No value: move GGML's own blocks into the backend's layout, "
             "bit-exactly, so no weight is rounded twice -- this only "
             "rearranges existing quantization, so it does nothing for a "
             "model stored as floats. "
             "'compact': the same with half-precision scales, 17%% smaller "
             "for a measured 8e-04 relative error. "
             "A width (2, 3, 4, 5, 6, 8): re-quantize to it, which works on "
             "any model -- 8 is imperceptible and the fastest option for an "
             "F16 file, 4 visibly reorders the top-5 tokens. A width is also "
             "the way to cap memory when the bit-exact path would round a "
             "narrow quantization *up*: a 3-bit i-quant has no exact affine "
             "form, so it is otherwise rebuilt at 4 bits and grows.",
    )
    p_run.add_argument(
        "--kv-bits", type=int, choices=(4, 8), metavar="BITS",
        help="Compress the key/value cache to this many bits. The weights "
             "are a fixed cost but the cache grows with the context, so at "
             "long prompts this is what decides whether one fits. Measured "
             "on a 1536-token context: 8 bits halves the cache for a 25%% "
             "slowdown and no visible change in output, 4 bits thirds it "
             "and starts to show. It costs speed rather than saving it.",
    )
    p_run.add_argument(
        "--experts", metavar="N",
        help="For a mixture-of-experts model with an index built by "
             "`reminis prepare`: how many experts to keep in memory. A "
             "number, or 'all' to hold the whole index and pin it there. "
             "Holding all of gpt-oss-20b's is 8.4 GB and takes it from 0.7 "
             "to 41 tok/s; a number smaller than the model runs it in less "
             "memory than it would otherwise need, more slowly.",
    )
    p_run.add_argument(
        "--draft", metavar="SOURCE",
        help="Speculative decoding: propose the next few tokens with "
             "something cheap and check them against this model in one "
             "batch. Decoding is memory-bound, so checking five tokens "
             "reads the weights once where producing five reads them five "
             "times, and the output is unchanged -- the accept/reject rule "
             "returns the target's own distribution. Either 'ngram', to "
             "draft from the context with no second model, or the path to "
             "a smaller model sharing this one's tokenizer, or "
             "'registry.db#name' to take the draft out of a registry.",
    )
    p_run.add_argument(
        "--draft-tokens", type=int, default=4, metavar="K",
        help="How many tokens --draft proposes each round (default 4). "
             "Too few wastes the batch; too many spends draft time on "
             "proposals a rejection earlier in the round throws away.",
    )
    p_run.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    p_run.set_defaults(func=cmd_run)

    # merge: combine several models into one
    p_hear = sub.add_parser(
        "transcribe",
        help="Transcribe audio with a speech model, out of the database",
        description="Whisper's encoder-decoder forward pass over tensors "
                    "selected from SQLite. The waveform, the spectrogram, the "
                    "two transformer stacks and the tokenizer all come out of "
                    "the one file -- no torch, no transformers, no ffmpeg.",
    )
    p_hear.add_argument("input", help="Path to the model database")
    p_hear.add_argument("audio", help="A PCM wav file. Any sample rate; "
                                      "resampled to 16 kHz and downmixed.")
    p_hear.add_argument("-n", "--max-tokens", type=int, default=224,
                        help="Most tokens to produce (default 224)")
    p_hear.add_argument("--language", default="en",
                        help="Language code to transcribe as (default en). "
                             "Ignored by the English-only checkpoints.")
    p_hear.add_argument("--task", choices=("transcribe", "translate"),
                        default="transcribe",
                        help="Transcribe in the spoken language, or translate "
                             "to English (default transcribe)")
    p_hear.add_argument("--temp", type=float, default=0.0,
                        help="Sampling temperature; 0 is greedy (default 0)")
    p_hear.add_argument("--seed", type=int, help="Seed, so a run repeats exactly")
    p_hear.add_argument(
        "--backend", choices=("auto", "numpy", "mlx", "cupy"), default="auto",
        help="Which array library to compute with. numpy is the reference "
             "implementation; mlx holds float16 and is faster.",
    )
    p_hear.add_argument("--tokens", action="store_true",
                        help="Print the token ids alongside the text")
    p_hear.add_argument("-q", "--quiet", action="store_true",
                        help="Print only the transcription")
    p_hear.set_defaults(func=cmd_transcribe)

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
    p_merge.set_defaults(func=cmd_merge)

    # registry: many models in one database
    p_reg = sub.add_parser(
        "registry",
        help="Keep many related models in one database, derived ones as deltas",
    )
    # `registry_parser` carries p_reg through to the handler, which prints its
    # help when `registry` is given with no subcommand.
    p_reg.set_defaults(func=cmd_registry, registry_parser=p_reg)
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
    p_log.set_defaults(func=cmd_log)

    # rollback: restore a model to a snapshot
    p_rollback = sub.add_parser(
        "rollback", help="Restore a model to its weights at a logged snapshot"
    )
    p_rollback.add_argument("log", help="Path to the training log database")
    p_rollback.add_argument("step", type=int, help="Snapshot step to restore")
    p_rollback.add_argument("-o", "--output", required=True, help="Output database path")
    p_rollback.set_defaults(func=cmd_rollback)

    # prepare: build the materialized expert index
    p_prepare = sub.add_parser(
        "prepare",
        help="Build a materialized index over a mixture-of-experts model's "
             "expert weights, so that running it is a seek rather than a "
             "decode",
    )
    p_prepare.add_argument("database", help="Path to the model database")
    p_prepare.add_argument(
        "--bits", type=int, default=4, choices=(3, 4, 6, 8),
        help="Width to store the indexed experts at (default 4)",
    )
    p_prepare.add_argument(
        "--group", type=int, default=128, choices=(32, 64, 128),
        help="Weights per scale in the index (default 128)",
    )
    p_prepare.add_argument(
        "--weights", action="store_true",
        help="Index the dense weight matrices instead of the experts, so "
             "that loading the model is a read rather than a decode. A "
             "quantized model is otherwise unpacked and re-packed in full "
             "on every run -- 466 seconds for a 27B at 3 bits, to rebuild "
             "what the last run already built. Costs a second copy of the "
             "weights on disk, which --drop reclaims.",
    )
    p_prepare.add_argument("--drop", action="store_true",
                           help="Delete the index and reclaim its space")
    p_prepare.set_defaults(func=_prepare)

    # quantize: write a quantized copy of a model
    p_quant = sub.add_parser(
        "quantize",
        help="Write a quantized copy of a model database (Q8_0 or Q4_0)",
    )
    p_quant.add_argument("database", help="Path to the model database")
    p_quant.add_argument("-o", "--output", required=True, help="Output database path")
    p_quant.add_argument(
        "--bits", type=int, default=4, choices=(4, 8),
        help="4 writes Q4_0 (about 28%% of float16), 8 writes Q8_0 (about "
             "53%%). Default 4. Both are real GGUF types, so the result "
             "exports back to GGUF and llama.cpp reads it.",
    )
    p_quant.set_defaults(func=cmd_quantize)

    # sweep: run one model at several precisions and compare
    p_sweep = sub.add_parser(
        "sweep",
        help="Run a model at several precisions and report what each costs in "
             "memory and in agreement with full precision",
    )
    p_sweep.add_argument("database", help="Path to the model database")
    p_sweep.add_argument(
        "--bits", default="8,6,4", metavar="LIST",
        help="Comma-separated widths to try (default 8,6,4)",
    )
    p_sweep.add_argument("--prompt", help="Text to measure agreement over. The "
                                          "default is a fixed mixed-domain passage.")
    p_sweep.add_argument(
        "--backend", choices=("auto", "numpy", "mlx", "cupy"), default="auto",
        help="Which array library to compute with (default auto)",
    )
    p_sweep.add_argument(
        "--kv-bits", type=int, choices=(4, 8), metavar="BITS",
        help="Also compress the key/value cache to this many bits",
    )
    p_sweep.add_argument(
        "--mix", action="store_true",
        help="Measure each group of tensors separately and derive a width per "
             "group for this model, instead of one width for all of it",
    )
    p_sweep.add_argument(
        "--budget", type=float, default=0.99, metavar="FRACTION",
        help="With --mix, the top-1 agreement a group must hold to keep a "
             "narrower width (default 0.99)",
    )
    p_sweep.set_defaults(func=cmd_sweep)

    # blame: when did a tensor last change?
    p_blame = sub.add_parser(
        "blame",
        help="Show when a tensor was updated and by how much",
        description="The transpose of `reminis log --step N`: given a tensor, "
                    "show every step that touched it.",
    )
    p_blame.add_argument("log", help="Path to the training log database")
    p_blame.add_argument(
        "param", nargs="?",
        help="Parameter name (or substring). Without it, lists all parameters.",
    )
    p_blame.add_argument(
        "--top", type=int, default=20,
        help="How many steps to show (default: 20, 0 for all)",
    )
    p_blame.set_defaults(func=cmd_blame)

    # bisect: binary-search for the step that broke something
    p_bisect = sub.add_parser(
        "bisect",
        help="Binary-search a training run for the step that broke something",
        description="Like git bisect: restore the midpoint snapshot, run a test, "
                    "exit code decides which half to keep. 0 = good, non-zero = bad, "
                    "125 = skip.",
    )
    p_bisect.add_argument("log", help="Path to the training log database")
    p_bisect.add_argument("--good", type=int, required=True, help="A known-good snapshot step")
    p_bisect.add_argument("--bad", type=int, required=True, help="A known-bad snapshot step")
    p_bisect.add_argument(
        "--test", required=True, metavar="CMD",
        help="Shell command to test a restored model. {db} is replaced with the "
             "path to the restored database. Exit 0 = good, 125 = skip, other = bad.",
    )
    p_bisect.set_defaults(func=cmd_bisect)

    # apply: apply a delta pack to a base model
    p_apply = sub.add_parser("apply", help="Apply a delta pack to a base model")
    p_apply.add_argument("base", help="Path to the base model database")
    p_apply.add_argument("delta", help="Path to the delta pack")
    p_apply.add_argument("-o", "--output", required=True, help="Output database path")
    p_apply.add_argument("--no-verify", action="store_true", help="Skip hash verification")
    p_apply.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    p_apply.set_defaults(func=cmd_apply)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # No subcommand: argparse leaves `func` unset, since only a subparser sets
    # it. Checking for the attribute rather than for `args.command is None`
    # means a subcommand added without a handler fails here, loudly, instead
    # of parsing successfully and silently doing nothing.
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args, parser)


def _prepare(args, parser):
    """Build or drop a materialized index -- of experts, or of weights."""
    from reminis import expert_index, packed_index

    if not Path(args.database).exists():
        parser.error(f"Database not found: {args.database}")

    index = packed_index if args.weights else expert_index
    label = "packed weight index" if args.weights else "expert index"

    if args.drop:
        before = Path(args.database).stat().st_size
        if not index.drop(args.database):
            print(f"There was no {label} to drop.")
            return
        after = Path(args.database).stat().st_size
        print(f"Dropped the {label} and reclaimed "
              f"{(before - after) / 1e9:.2f} GB.")
        return

    if args.weights:
        def weight_progress(done, total, name, written, elapsed):
            rate = written / elapsed / 1e9 if elapsed else 0
            print(f"\r  [{done:>3}/{total}] {name:<32} "
                  f"{written / 1e9:5.2f} GB, {rate:.2f} GB/s",
                  end="", flush=True)

        print(f"Packing the weights of {args.database} at {args.bits} bits")
        summary = packed_index.build(args.database, bits=args.bits,
                                     group_size=args.group,
                                     progress=weight_progress)
        print()
        print(f"Wrote {summary['tensors']:,} tensors, "
              f"{summary['bytes'] / 1e9:.2f} GB at {summary['bits']} bits, "
              f"in {summary['seconds']:.0f}s.")
        print(f"`reminis run --pack {args.bits}` will now read them straight "
              f"out of it. `reminis prepare --weights --drop` undoes this.")
        return

    def progress(done, total, name, experts, written, elapsed):
        rate = written / elapsed / 1e9 if elapsed else 0
        print(f"\r  [{done:>3}/{total}] {name:<32} {experts:>5} experts, "
              f"{written / 1e9:5.2f} GB, {rate:.2f} GB/s",
              end="", flush=True)

    print(f"Indexing the expert weights of {args.database}")
    summary = expert_index.build(args.database, bits=args.bits,
                                 group_size=args.group, progress=progress)
    print()
    print(f"Wrote {summary['experts']:,} experts from "
          f"{summary['tensors']} tensors, {summary['bytes'] / 1e9:.2f} GB at "
          f"{summary['bits']} bits, in {summary['seconds']:.0f}s.")
    print("`reminis run` will now read experts straight out of it. "
          "`reminis prepare --drop` undoes this.")


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


def _show_blame(log_path: str, param: str | None, top: int):
    """Show every step that touched a parameter."""
    from reminis.track import TrainingLog

    if not Path(log_path).exists():
        print(f"Error: {log_path} not found")
        sys.exit(1)

    log = TrainingLog(log_path)
    try:
        if param is None:
            names = log.param_names()
            print(f"Parameters in {log_path} ({len(names)}):\n")
            for name in names:
                count = log.conn.execute(
                    "SELECT COUNT(*) FROM param_updates WHERE param = ?", (name,)
                ).fetchone()[0]
                print(f"  {name:<60s} {count:>5} steps")
            return

        names = log.param_names()
        exact = [n for n in names if n == param]
        if not exact:
            matches = [n for n in names if param in n]
            if not matches:
                print(f"No parameter matching '{param}'.")
                print(f"List all with: reminis blame {log_path}")
                sys.exit(1)
            if len(matches) > 1:
                print(f"'{param}' matches {len(matches)} parameters:\n")
                for m in matches:
                    print(f"  {m}")
                print(f"\nNarrow it down or use the full name.")
                return
            param = matches[0]

        rows = log.param_history(param)
        if not rows:
            print(f"No updates recorded for '{param}'.")
            sys.exit(1)

        snap_steps = set(log.snapshot_steps())

        print(f"Parameter: {param}")
        print(f"Updated at {len(rows)} steps\n")
        print(f"  {'STEP':>6}  {'|grad|':>10}  {'mean':>10}  {'max':>10}  "
              f"{'|w| before':>10}  {'|w| after':>10}  {'':>4}")
        print("  " + "-" * 72)

        display = rows if top == 0 else rows[-top:]
        if top and len(rows) > top:
            print(f"  ... ({len(rows) - top} earlier steps omitted, use --top 0 for all)")

        for step, gnorm, gmean, gmax, wbefore, wafter, rolled in display:
            snap = "snap" if step in snap_steps else ""
            rb = " [rolled back]" if rolled else ""
            print(
                f"  {step:>6}  {gnorm:>10.4f}  {gmean:>10.6f}  {gmax:>10.4f}  "
                f"{wbefore if wbefore else 0:>10.4f}  {wafter if wafter else 0:>10.4f}  "
                f"{snap}{rb}"
            )

        biggest = max(rows, key=lambda r: r[1])
        print(f"\n  Largest gradient at step {biggest[0]} (|grad| = {biggest[1]:.4f})")
        if snap_steps:
            print(f"  Snapshots at: {sorted(snap_steps)}")
    finally:
        log.close()


def _run_bisect(log_path: str, good: int, bad: int, test_cmd: str):
    """Binary-search through snapshots for the step that broke something."""
    import shutil
    import subprocess
    import tempfile

    from reminis.track import TrainingLog, rollback_to_step

    if not Path(log_path).exists():
        print(f"Error: {log_path} not found")
        sys.exit(1)

    log = TrainingLog(log_path)
    try:
        all_snaps = log.snapshot_steps()
        if not all_snaps:
            print("No snapshots in this log.")
            sys.exit(1)

        if good not in all_snaps:
            print(f"No snapshot at step {good}. Available: {all_snaps}")
            sys.exit(1)
        if bad not in all_snaps:
            print(f"No snapshot at step {bad}. Available: {all_snaps}")
            sys.exit(1)

        lo_good = min(good, bad)
        hi_bad = max(good, bad)
        candidates = [s for s in all_snaps if lo_good <= s <= hi_bad]

        if len(candidates) < 2:
            print(f"Need at least two snapshots between good ({good}) and bad ({bad}).")
            sys.exit(1)

        print(f"Bisecting between step {lo_good} (good) and step {hi_bad} (bad)")
        print(f"{len(candidates)} snapshots in range: {candidates}\n")

        lo_idx = 0
        hi_idx = len(candidates) - 1

        tmpdir = Path(tempfile.mkdtemp(prefix="reminis-bisect-"))
        try:
            while hi_idx - lo_idx > 1:
                mid_idx = (lo_idx + hi_idx) // 2
                mid_step = candidates[mid_idx]
                remaining = hi_idx - lo_idx - 1
                print(f"  Testing step {mid_step} ({remaining} steps left to search)...")

                db_path = str(tmpdir / f"bisect-{mid_step}.db")
                rollback_to_step(log, mid_step, db_path, verbose=False)

                cmd = test_cmd.replace("{db}", db_path)
                result = subprocess.run(cmd, shell=True)
                Path(db_path).unlink(missing_ok=True)

                if result.returncode == 0:
                    print(f"    step {mid_step}: good")
                    lo_idx = mid_idx
                elif result.returncode == 125:
                    print(f"    step {mid_step}: skip")
                    candidates.pop(mid_idx)
                    hi_idx = min(hi_idx, len(candidates) - 1)
                else:
                    print(f"    step {mid_step}: bad (exit {result.returncode})")
                    hi_idx = mid_idx

            first_bad = candidates[hi_idx]
            last_good = candidates[lo_idx]
            print(f"\n  First bad snapshot: step {first_bad}")
            print(f"  Last good snapshot: step {last_good}")
            if first_bad - last_good > 1:
                print(f"  (the actual regression is somewhere between "
                      f"step {last_good} and step {first_bad} -- snapshots "
                      f"are not per-step, so this is as precise as it gets)")
            print(f"\n  Inspect the bad step:  reminis log {log_path} --step {first_bad}")
            print(f"  Restore the good one: reminis rollback {log_path} {last_good} -o restored.db")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        log.close()
