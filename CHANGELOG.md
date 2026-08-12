# Changelog

Measurements are on `Llama-3.2-1B-Instruct-f16` (2.5 GB, 147 tensors) unless
another model is named, taken on an M-series Mac with 17 GB of memory. Where a
release reports a before and an after, both were measured on the same machine
in the same cache state -- the earlier code was checked back out and re-run
rather than compared against a number from earlier in the day.

`bench.py` produces these figures; `tests/test_prerelease.py` is the gate every
release below had to pass.

## 0.32.3 -- each command next to the flags it reads

`main` was 495 lines: roughly 350 building 24 subparsers, then 140 of
`elif args.command == "..."` dispatching to them. Every command's argument
handling sat three hundred lines from the flags it consumed.

Each command is now a `cmd_*(args, parser)` function attached to its subparser
with `set_defaults(func=...)`, and `main` is fourteen lines that parse and
call. `parser` is passed because a handler's remaining job is usually to
reject a combination argparse cannot express, and `parser.error` is what
prints usage and exits 2 the way every other argparse failure does.

| | before | after |
| --- | --- | --- |
| `main` | 495 lines | **14** |
| longest function in `cli.py` | 495 | 371 (`build_parser`, declarative) |

Nothing about the interface moved. Every subcommand's `--help`, both parsers'
usage, all six argument-validation errors, the bare-command and unknown-command
paths and their exit codes were captured before the change and compared after:
byte-identical across 32 cases. `convert`, `info`, `quantize`, `export`,
`view`, `diff`, `apply`, `run`, `prepare` and the five `registry` subcommands
were also run end to end against a real model.

A subcommand with no branch used to parse successfully and then do nothing.
`main` now dispatches on the `func` attribute, so that case prints help and
exits 1, and the pre-release gate asserts every subcommand reaches a callable
handler with the right signature -- checked against a deliberately unwired
command.

## 0.32.2 -- the viewer's numbers, and one place that opens a database

**Bug fix: the viewer reported wrong statistics for every BF16 model.** It
decoded tensor bytes with its own `np.frombuffer` table rather than through
`dtypes.to_float32`, and that table read BF16 as float16. The two are both 16
bits wide, so nothing raised -- a stored 100.0 was reported as 3.39, and every
mean, standard deviation, min, max and heatmap cell of a BF16 model was wrong.
Safetensors models are predominantly BF16.

`tests/test_safetensors.py` already carried a comment warning that reading
BF16 as float16 is silent rather than fatal; the viewer had re-implemented the
decode by hand and walked into exactly that. Both call sites now go through
`to_float32`, which also fixes F64 tensors, previously reported as *quantized*
with no statistics at all.

The pre-release check for this asserted the HTML file existed and was over
1000 bytes. It now stores known values as F32, F16, BF16 and F64 and asserts
the reported minimum, maximum, mean, zero fraction, element count and heatmap
cells come back exactly. Against the old code it fails 11 checks.

### One place that decides how a database is opened

Thirty `sqlite3.connect` calls and twenty-eight loose `PRAGMA` lines, so a
setting measured on one path never reached the others. They now come from
`db.py`, which names the four reasons reminis opens a file -- `open_for_bulk_write`
for a fresh one, `open_for_bulk_copy` for a file copied moments earlier,
`open_for_append` for a registry or training log, `open_read_only` for anything
that must not be modified -- and records against each why its pragmas differ.

Three callers had been left behind by earlier releases. The pack written by
`lora`, the snapshot written by `track` and the export written by `registry`
all unlink their target and write it whole, which is the case 0.27.0 tuned in
the converter and 0.29.1 tuned for delta packs. All three still went through
WAL with a rollback journal, and each `INSERT` still committed on its own:

| writing 2472 MB into a fresh file | before | after |
| --- | --- | --- |
| WAL, `synchronous=NORMAL`, implicit commits | 7.15s (346 MB/s) | -- |
| journal off, 64 KB pages, one transaction | -- | **1.12s (2209 MB/s)** |

Best of three alternating runs; the spread was 7.15-7.41s against 1.12-1.41s.

`merge` deletes a failed output rather than leaving it. It writes into a copy
of its first input with no journal, so a merge that raises partway would
otherwise leave a half-written file that reads like a finished one. Verified
by failing a merge deliberately.

No behaviour change beyond that: the same bytes are written, and the merge,
lora, track, registry, diff, quantize, sweep, lowrank and blame/bisect suites
all pass unchanged.

### One definition of what a tensor is

The tensor columns were declared in two schemas and written by eight separate
`INSERT` statements, each spelling out its own placeholder list. That is the
shape of the bug fixed in 0.29.0, where a pack insert named nine columns and
bound eight and nothing caught it, because the only path reaching it added a
tensor rather than changing one.

Both now come from `TENSOR_COLUMNS` in `db.py`. The DDL shares a column block
between the single-model schema and the registry's, which differ only in
uniqueness -- `name` against `(model_id, name)` -- and the four statements
(`INSERT_TENSOR`, `INSERT_OR_REPLACE_TENSOR`, `UPSERT_TENSOR`,
`INSERT_REGISTRY_TENSOR`) are generated, so a placeholder list cannot disagree
with a column list.

Verified structurally rather than by eye: both schemas were built from the
released code and from this one and compared through `table_info`,
`index_list`, `index_info` and `foreign_key_list`. Identical.

The pre-release gate now prepares each statement against a real schema and
asserts `TENSOR_COLUMNS` is the table's column list, so a count mismatch or a
column added to one side only fails at release rather than on the one code
path that happens to reach it.

The 18 distinct `SELECT ... FROM tensors` shapes were left alone. They select
different subsets for good reasons, and routing them through a builder would
trade readable SQL for indirection without removing a failure mode.

## 0.32.1 -- README says what is now guaranteed

The round-trip tables reported an exported file smaller than the one that went
in -- 258.3 MB against 256.7 MB on SmolLM-135M f16 -- and that gap was the
tokenizer going missing. The evidence was published in the README for the life
of the project and nothing compared the two columns. Regenerated: they match
exactly now, and the wording says the guarantee covers metadata rather than
only weights.

Closes the last item of #11. The safetensors export path was checked for the
same fault and does not have it: it stores config values as JSON rather than a
Python repr, so lists and nested dicts survive untouched.

## 0.32.0 -- exported models keep their tokenizer

`reminis export` dropped every array-typed metadata field, and had since
arrays were first skipped:

```python
elif type_name == "array":
    pass  # arrays are complex; skip for now to avoid corruption
```

For a llama-family model that is the tokenizer. An exported GGUF carried
correct weights, no vocabulary and no merges, and llama.cpp refused it with
`error loading model vocabulary: cannot find tokenizer merges in model file`.
Found by running a full cycle -- convert, generate, export, generate again --
rather than by any test.

Arrays are now written back. The element type had nowhere to live, since a
stored `[[1], [1]]` reads the same whether it was int32 or uint32, so the
`type` column now records it as `array:int32`, `array:string` and so on. A
database written before this carries a bare `array` and still exports, with
the element type inferred from the values -- exact for strings and floats,
and int32 for whole numbers.

On SmolLM-135M the exported file is now byte-identical to the file it was
converted from, and llama.cpp generates the same text from both.

**The tests said this was fine.** `test_roundtrip.py` compared tensor hashes
and passed 22/22 while every one of those files had lost its tokenizer; the
pre-release GGUF check had inherited the same blind spot. Both now compare
metadata field by field, and the pre-release one asserts the exported file is
byte-identical. Checked against the old code, where they fail on exactly the
five dropped fields while the tensor check still reports success.

## 0.31.0 -- streaming GGUF export

Exporting wrote every tensor into the `gguf` writer and only then flushed the
file, which held the whole model in memory and pushed each tensor through
numpy's `tofile` on a buffered file object. `add_tensor_info` registers the
same header entry without the data, so the header is laid out first and the
weights stream past it one at a time.

| | before | after |
| --- | --- | --- |
| `export` gpt-oss-20b (12.1 GB) | 74.3s | **5.2s** |
| peak memory | 3.85 GB | **1.61 GB** |

Only visible at scale -- the same path on a 1B model looked healthy. Verified
byte-identical against the previous implementation across F16, Q4_K_M, Q2_K and
mixed-dtype models, with the full round-trip suite passing on all 22 models.

Also fixed `bench.py`'s `weights_hash`, which divided by file size and so
reported nearly twice the true rate on any model carrying an expert index.

## 0.30.0 -- threaded delta encoding, and a benchmark that meant something

`bench.py`'s `diff_changed` used a quantized target. Its blobs are a different
size from the base's, so encoding returned after one compression and never
reached XOR or the bit-plane split -- 0.78s where a same-dtype fine-tune, the
case delta packs exist for, costs 6.3s. The benchmark now perturbs mantissa
bits in place.

On that workload each tensor is compressed three or four ways so the smallest
can be kept, and those now run concurrently.

| | before | after |
| --- | --- | --- |
| encode phase | 6.69s | **4.45s** |
| `diff` against a fine-tune | 10.76s | **8.94s** |

Bandwidth-bound rather than core-bound: two threads and four threads both give
1.50x, and end to end -- where two threads are already hashing -- four is worse
than two. `ZstdCompressor` is not thread-safe and does not fail when shared; it
returns different bytes. Compressors are now per-thread, and every payload is
checked against the single-threaded encoder.

## 0.29.1 -- delta packs written like every other new file

A pack is unlinked and recreated immediately before it is opened, but was still
written through WAL with `synchronous=NORMAL`. Its commit alone was 1.4s of a
3.7s diff.

| | before | after |
| --- | --- | --- |
| `diff` with 113 changed tensors | 3.74s | **1.68s** |

## 0.29.0 -- hashing a model while comparing it

`diff` read both models to compare them and then read both again to hash them
for the pack. The hashes are now accumulated during the comparison, and each
runs on its own thread so the two overlap each other and the reading.

| | before | after |
| --- | --- | --- |
| `diff` of identical models | 2.78s | **0.95s** |
| `diff` with 113 changed tensors | 5.47s | **3.74s** |

Inline accumulation is only valid when both models hold the same tensors, since
the hash is defined over every tensor in name order and the loop walks the
intersection. Differing tensor sets fall back to hashing each model separately,
now concurrently.

**Bug fix, present since delta packs existed:** the insert for a tensor found
only in the target named nine columns and bound eight, so any diff that *added*
a tensor failed with `8 values for 9 columns`. Nothing exercised that path.

## 0.28.0 -- quantizing in slices, across cores

Seventy percent of quantization was block arithmetic, and most of that went to
temporaries: `astype(float32)` copying a tensor that was already float32,
`np.abs(blocks).max()` materialising a full-size absolute value for one
reduction, and a fresh array for each of rint, clip and the +8. Q8_0 now takes
its per-block extreme as `max(max, -min)` and rounds in place. The remaining
work is split into 64k-block slices across up to eight threads, which also caps
the working set -- an embedding matrix that wanted a gigabyte of float32 at
once now moves in 8 MB pieces.

| | before | after |
| --- | --- | --- |
| `quantize --bits 8` | 4.70s | **1.60s** |
| `quantize --bits 4` | 4.87s | **1.44s** |

Output is byte-identical. Q4_0 still selects its extreme with `argmax` over
`|x|` rather than comparing `max` against `-min`: the two disagree when a
block's largest positive and largest negative values have equal magnitude, and
`argmax` is what every earlier version wrote.

## 0.27.1 -- no journal for a file that was a copy a moment ago

`apply_delta` wrote its result through a rollback journal, so every
reconstructed tensor was written twice and fsynced. The database being
protected is a copy taken seconds earlier in the same call, and a failed apply
is deleted and rerun.

| | before | after |
| --- | --- | --- |
| write phase (1.3 GB delta) | 7.05s | **1.79s** |
| `apply` end to end | 12.6s | **7.4s** |

No raised `cache_size` here, deliberately: the converter raises it because it
builds a file from nothing, and raising it on this path measured slower.

## 0.27.0 -- reading a GGUF header without a numpy view per string

`GGUFReader` builds two memory-mapped numpy views for every string in the file.
A 128k-token vocabulary is 400k of them, and on Llama-3.2-1B that alone was
7.3s of a 19.9s conversion. The same bytes read with `struct`, and one
`np.frombuffer` per numeric array, take 0.18s.

| | before | after |
| --- | --- | --- |
| `convert` from GGUF | 19.9s | **1.9s** |

The parser is checked against the reference reader on all 20 GGUF models here,
field for field and tensor for tensor, and anything it does not recognise falls
back to that reader rather than being guessed at.

Writing is now one transaction with the journal off -- which protects nothing
on a file that did not exist a moment ago -- and 64 KB pages. The page size
outlives the connection, so it was chosen by measuring reads too: against
SQLite's 4 KB default it is 5.5x the write throughput, 3.6x sequential read,
and 2.8x on the small random byte-range reads the expert index does.
Safetensors import and `quantize` write fresh databases the same way.

## 0.26.1 -- parallel apply, and a pre-release gate

`apply_delta` now overlaps the base-model hash with the file copy, decodes
deltas on a thread pool, and pipelines the result hash through a reader thread.

| | before | after |
| --- | --- | --- |
| `apply` end to end | 41s | **12.6s** |

`ZstdDecompressor` was a module-level singleton that corrupted its output under
concurrent use; it is now per-thread.

Adds `tests/test_prerelease.py`, a gate covering conversion, diff, apply,
quantize, merge, registry, viewer, low-rank packs, the bit-plane encoding and
inference against SmolLM-135M. It grew from 38 checks to 71 over the releases
above. The existing suite round-trips every model in `models/` and trains real
networks, which is thorough and takes hours; this runs in about twenty seconds.

Adds `bench.py`, which measures the paths these releases change.

---

## Where things stand

On gpt-oss-20b (12.1 GB of tensors, plus a 9 GB expert index):

| | time | throughput |
| --- | --- | --- |
| `sqlite_blob_read` | 4.25s | 2845 MB/s |
| `export` to GGUF | 5.22s | 2316 MB/s |
| `weights_hash` | 5.99s | 2021 MB/s |
| `diff`, identical | 8.45s | -- |
| `convert` from GGUF | ~18-22s | ~600-700 MB/s |

The storage paths are within roughly 1.6x of a hard limit. `convert` is the
slowest and was investigated rather than optimised: reads run at 6846 MB/s and
are not the constraint, and SQLite ingests large blobs at 600-700 MB/s against
a plain file write plus fsync of 1041 MB/s on the same machine -- about 60% of
raw disk, which is what B-tree overflow chains cost. Five pragma variants (page
size, cache size, exclusive locking) produced nothing that survived repeated
runs.

One caveat on the export figure: it does not fsync, so 2316 MB/s is partly
memory rather than sustained disk. The comparison against 74.3s is still fair,
since the old path did not fsync either.
