# Changelog

Measurements are on `Llama-3.2-1B-Instruct-f16` (2.5 GB, 147 tensors) unless
another model is named, taken on an M-series Mac with 17 GB of memory. Where a
release reports a before and an after, both were measured on the same machine
in the same cache state -- the earlier code was checked back out and re-run
rather than compared against a number from earlier in the day.

`bench.py` produces these figures; `tests/test_prerelease.py` is the gate every
release below had to pass.

## 0.34.0 -- a speech model, and a 7B that repacks in 31 s

### Whisper runs out of the database

`reminis transcribe` is a second forward pass. Whisper is an encoder-decoder
over audio and shares almost nothing with the block `reminis run` implements,
so it is its own module rather than an entry in `arch.py` -- that registry
holds *deviations* from a shared block, and here there is no shared block
left.

```bash
reminis convert ./whisper-tiny/ -o whisper.db
reminis transcribe whisper.db speech.wav
```

The waveform, the spectrogram, both transformer stacks and the tokenizer all
come out of the one file: no torch, no `transformers`, no ffmpeg, no config on
disk. Against `transformers` in float32 on identical features -- encoder
hidden states at correlation 1.00000000 (1.0e-04 relative), decoder logits at
1.00000000 (2.0e-06), the top-5 identical in order, and the greedy
transcription **token for token identical**. whisper-base and whisper-tiny.en
work with no code change, because the layer counts, widths and token ids are
read rather than assumed.

That last one nearly did not. **The English-only checkpoints number every
special token one lower** -- `.en` starts its decoder at 50257 and marks
no-timestamps with 50362, where the multilingual models use 50258 and 50363 --
because they carry no language or task tokens. A constant for either family
primes the other one's decoder with the wrong prefix, and that transcribes
fluently and wrongly rather than failing. The prefix is now resolved by name
against the vocabulary in the database, which is stable where a number is not.

Six things had no counterpart in the rest of reminis, and each one fails by
producing a fluent English sentence that is not what was said: an 80-channel
Mel spectrogram (Slaney scale *and* Slaney normalisation -- both have a second
convention in circulation); two 1-D convolutions, the second strided;
LayerNorm rather than RMSNorm; a gate-free GELU feed-forward with exact `erf`;
cross-attention over the audio, whose keys and values are computed once for
the whole transcription; and Whisper's two lists of suppressed tokens, which
were the only thing still wrong once the logits already matched to 2e-06.

Two failures are recorded because both looked like working models:

**Float16 breaks LayerNorm quietly.** Its statistics are a sum over the whole
channel -- 384 squared activations of magnitude ~17 reach six figures, and
float16 saturates at 65504 and returns infinity. The encoder still correlated
at **0.956** with the reference. The statistics are now taken in float32
whatever the weights are stored as.

**The backends take a boolean mask, not an additive one.** `attention` wants
true where a query may see a key; additive `-inf` inverts it, every row masks
itself out, and the softmax returns NaN. That one was loud only because NaN
propagates.

### And it is the fastest of the tools tried

Same clip, whisper-tiny, greedy, everyone handed the same samples and
computing their own spectrogram. Interleaved, nine rounds, medians:

| Engine | Warm | xRT | Range |
|---|---|---|---|
| **reminis (mlx)** | **0.108 s** | **192x** | [0.104-0.113] |
| mlx-whisper | 0.118 s | 176x | [0.117-0.122] |
| openai-whisper | 0.322 s | 64x | [0.319-0.355] |
| faster-whisper int8 | 0.435 s | 48x | [0.428-0.543] |
| transformers | 0.519 s | 40x | [0.514-0.526] |

Ahead by 9%, with ranges that do not overlap -- which is the only reason a
margin that size is worth stating. **Cold it loses**, 0.142 s against
~0.118 s, and that is the storage layer rather than the forward pass:
reminis reads a 154 MB float32 database and narrows it where mlx-whisper
memory-maps 75 MB of float16.

It started at 0.656 s, slower than everything. The 6.1x came from six
changes, and the first three are accounting errors rather than
optimizations -- **65% of a transcription was spent parsing a vocabulary**:

- the tokenizer was rebuilt on every call: 266 ms, against 0.04 ms to decode
- `prompt_tokens` re-parsed the whole vocabulary, three times: 236 ms
- the vocabulary was stored as a Python `repr` and read with
  `ast.literal_eval`; as JSON it is 82 ms -> 4.6 ms, and `_parse_array`
  now tries JSON first so older databases keep working
- three LayerNorms per layer at eleven operations each became
  `mx.fast.layer_norm`: 1.64 ms a token -> 1.10 ms
- greedy decoding does its argmax on the device, so one integer crosses the
  boundary per token rather than a 51,865-wide row
- prefill projects only the last position through the vocabulary

The phase that looked expensive was innocent. "Prefill" measured 245 ms and
turned out to cost 3 ms; the time belonged to `prompt_tokens`, called on the
line above it and swept into the same window.

**Fusing Q, K and V measured worse and was reverted.** It is worth a third
in the text path. Here it took a transcription from 137 ms to 220 ms: these
projections are 384x384 and a decoded token is one row, so they are
dispatch-bound, and three slices plus a reshape of each strided view cost
more than the two dispatches saved.

### A safetensors conversion now carries its tokenizer

A GGUF holds its tokenizer inside the file; safetensors holds none, so such a
database could compute correct logits and not say which words they stood for.
A sibling `tokenizer.json` is now ingested into `tokenizer.ggml.*` and read by
the existing byte-level BPE implementation. Only BPE is taken: claiming a
vocabulary reminis cannot drive would be worse than leaving it out, because a
wrong tokenizer decodes to fluent nonsense.

This is the same completeness gap that dropped array metadata on export until
0.32.0, found in the other direction.

### Kokoro is stored, verified, and not run

`hexgrad/Kokoro-82M` converts and exports **byte-identical, 548/548 tensors
SHA256-matched**, and its structure is a `GROUP BY` rather than an
archaeology exercise. It is not runnable here, and the reason is specific
rather than a pending roadmap item: two thirds of it is an iSTFTNet vocoder,
89 of its tensors are weight-normalised `weight_v`/`weight_g` pairs, and it
needs `espeak-ng` for phonemes. None of that is a transformer, so none of the
machinery Whisper reused applies.

### Reads go wide, because SQLite lets them

Export alternated a blocking read with a write, so neither the disk nor the
writer was ever busy. SQLite serializes writers and does **not** serialize
readers, so the read half parallelises even where the write half cannot.

Measured on a 4.4 GB model, reading every tensor by name:

| Readers | Throughput |
|---|---|
| 1 thread | 3570 MB/s |
| 4 threads | 15079 MB/s |
| 8 threads | 16056 MB/s |

`db.read_blobs_ahead` is the shared form: one connection per thread, a byte
budget rather than a queue depth -- one tensor can be half a gigabyte -- and
**order preserved**, because an export has already written the offsets its
tensors must land at.

| | before | after |
|---|---|---|
| GGUF export | 1621 MB/s | **3347 MB/s** |
| Weights hash | 1120 MB/s | **1700 MB/s** |
| safetensors export | same loop | same fix |

Output is byte-identical and the digest is unchanged.

**Hashing each tensor separately would reach 7 GB/s, and is not done.** It
would be a different number, and these digests are written into delta packs
and checked by `apply`, so changing them would make every pack ever written
unopenable. The reads get the threads; the digest keeps its order.

**Conversion was measured and left alone.** At 895 MB/s it looked like the
slow half, but a bare SQLite blob insert of the same bytes runs at 892 MB/s
-- it is already at the storage engine's ceiling, and the memoryview it
binds is doing its job, since binding `bytes` instead measures the same. The
gap to a raw file copy (3930 MB/s) is SQLite's page and B-tree overhead, not
reminis's.

### Repacking a 7B at load: 83 s -> 31 s

Three changes, measured separately on Mistral-7B Q4_K_M with a warm page
cache.

**The prefetch was silently discarding every affine repack.** `_unpack_one`
referenced `arr.nbytes` on the repacked path, where `arr` does not exist; the
blanket `except Exception` ate the `NameError`, so every Q4_K, Q5_0 and Q8_0
tensor fell through to the synchronous path and six worker threads did
nothing at all. On a Q4_K_M model that is almost the whole load.

**The bit packing was expanding every value to individual bits.**
`_pack_words` went through `unpackbits`/`packbits`, which was 89% of a single
tensor's repack. For widths that divide 32 -- 4 and 8, the common cases -- it
is now a vectorised shift-and-OR: **4.6x faster**. The bit-stream path stays
for the 5-bit types, which straddle word boundaries.

**The main thread was racing the workers it was waiting for.** With nothing
to coordinate them, `get` did the work itself whenever a worker had not
finished, and it kept winning: 21 prefetch hits against 726 misses. It now
waits on a condition variable for a tensor already in flight -- **226 hits
against 65**.

| Mistral-7B Q4_K_M, `--pack compact` | before | after |
|---|---|---|
| Time to first token | 85 s | **31 s** |
| Processor utilisation | ~100% | ~245% |

Granite-3.1-1b-a400m came down from 13.9 s to 4.3 s on the same changes.

## 0.33.1 -- the index says which width it is, and `--chat` finishes the turn

Three things 0.33.0 got wrong quietly, and one measurement that came back
negative and is recorded rather than shipped.

**A packed index now says when it is not the width you asked for.** It holds
one width, and a run asking for another got the index's silently -- so
`--pack 4` against an index built at 3 could produce three-bit weights and
report nothing. The index still wins, because rebuilding every weight per run
is the cost it exists to remove, but it says so and names the command that
changes it.

**`--chat` finishes the assistant turn on a reasoning model.** These templates
do not stop at `<|im_start|>assistant\n`; they open a thinking channel too,
and the model was trained expecting to find one open. Left off, the model
opens its own and reasons for as long as it likes, so any ordinary token
budget showed working and no answer -- which reads exactly like a broken
forward pass. `--chat` now closes the channel immediately, which is how these
templates express "answer directly", and `--think` leaves it open for when the
working is the point.

**The packed index has tests.** It shipped with none. They check the property
that matters: logits through the index are *bit-identical* to logits without
it, the index is demonstrably the thing that served them rather than assumed
to be, a float run ignores it, and building then dropping leaves the model
byte-for-byte where it started.

**The bit-exact repack now threads.** `ggml_repack` moves bits rather than
decoding, and the prefetch skipped it on that reasoning -- which left a
bit-exact load of a Q4_K model entirely single-threaded. Qwen3.5-4B, cold, to
first token: **66s -> 58s**. Eight threads repack 3.8x faster in isolation and
this is 12%, because the pipeline is bounded by the device upload the main
thread does either way. The isolated figure is the one to distrust.

**Compiling the recurrence was tried and removed.** `mx.compile` over the
whole delta-rule step is the obvious move -- a dozen small elementwise
operations issued one at a time, forty-eight layers deep on the 27B -- and it
measured 25.29 tok/s against 25.07 on the 4B, and 4.36 against 4.6 on the 27B.
Neither is a difference. What the step spends is in the two matrix products,
which were already single kernels. The finding is a comment in `arch.py` so
the next person does not spend the afternoon on it.

## 0.33.0 -- a hybrid architecture, and loading it as a read

**New architecture: `qwen35`,** which Qwen 3.5 and Qwen 3.8 are written in.
Three layers in every four are Gated DeltaNet -- a recurrent block carrying a
matrix-valued state rather than a key/value cache -- and the fourth is
attention with a learned gate on its output. A 27B model is 48 recurrent
layers and 16 attending ones.

The recurrence is a *delta* rule, and that is the part worth stating plainly
because writing it as the more obvious thing produces a model that talks:

    S = exp(gate) * S                decay what is stored
    S = S + k (v - S^T k)^T beta     write only the difference
    y = S^T q / sqrt(head_dim)

Accumulating `k v^T` instead of the correction looks nearly identical and
behaves nothing like it -- a key that recurs keeps adding a value the state
already holds, the state grows along that direction until it dominates every
readout, and the model emits one token forever. No error, no NaN. Two further
details have the same character: value heads pair to key heads by tiling
(`h % n_k_heads`) rather than by repeating, and the readout carries a
`1/sqrt(head_dim)` the state does not. Each was found by checking every
intermediate tensor of layer 0 against llama.cpp's own trace; all twenty-four
now agree to within float noise.

Two smaller things this model needed, both of which were silently wrong
before rather than absent:

* its pre-tokenizer, which splits digits one at a time where the fallback
  took whole runs -- `2026` is four tokens here and one under GPT-2's rules,
  so every number in every prompt was being handed over as ids the model was
  never trained on;
* `USER_DEFINED` tokens matched whole. The SentencePiece tokenizer already
  took token types 3 and 4; the byte-level one took only 3, so `<think>` was
  shattered into pieces that BPE could not rebuild. The three reference
  tokenizers this suite checks against are unaffected by the fix.

**New: a packed index over dense weights.** `expert_index` exists because a
mixture of experts re-derives the kernel's layout per token. A dense model has
the same problem at load: it unpacks and re-packs every weight before the
first token and keeps none of it, so the next run does it again. On
Qwen3.8-27B at 3 bits that was 466 seconds to rebuild 11 GB the previous run
had already built.

    reminis prepare model.db --weights --bits 3
    reminis run model.db --pack 3
    reminis prepare model.db --weights --drop

Loading becomes a primary-key seek and a memcpy into the layout the quantized
matmul already wants. Time to first token, Qwen3.8-27B: **466s -> 5.3s**. It
costs a second copy of the weights on disk -- 11.07 GB beside the original
9.85 GB -- which is the trade an index always offers, and `--drop` reclaims
it. The original tensors are untouched, so the database is still the model
and still converts back.

Decoding got faster too, which was not the point and is worth explaining: the
index never materialises the float32 a quantization is unpacked through, so
peak memory falls far enough that the system stops compressing the weights
and faulting them back in mid-forward-pass. Qwen3.8-27B went from 1.0 to
**4.6 tok/s** on a 16 GB machine.

**Two general speed fixes,** both applying to any packed model on Apple
silicon:

* the carried state of a recurrent layer is scheduled rather than waited on.
  It has to be collapsed every token or the graph grows without bound, but
  nothing needs its value until the *next* token, so blocking stalled the
  processor against the device once per layer -- forty-eight times a token on
  the 27B.
* dense weights are wired, so the system cannot compress them. `reserve` has
  done this for the expert index since it was written, with a measurement
  attached; a dense model was never given the same treatment and has the same
  problem for the same reason.

Qwen3.5-4B, decode: **13.6 -> 25.1 tok/s**, against llama.cpp b10270's 17.7
on the same file and machine.

`--pack` now takes 2, 3 and 5 as well as 4, 6 and 8. A three-bit i-quant has
no exact affine form, so the bit-exact path rebuilds it at four bits and it
*grows*; naming the width is how a model that only just fits is made to fit.

## 0.32.4 -- an export that cannot write a field stops

**Bug fix: `export` silently dropped metadata it could not encode.** The loop
writing metadata to a GGUF wrapped each field in `except Exception: pass`, so
a field that failed to encode vanished and the export still reported success.
That is precisely how 0.32.0 shipped models with no tokenizer -- and this
would have hidden that bug coming back.

A second path was quieter still: `_write_meta_value` recognised a fixed set of
type names and an unrecognised one fell off the end of the function, writing
nothing, while the caller counted it as written. So `Wrote N metadata fields`
could name fields that never reached the file.

Both are now loud. An unknown type raises and names the field, and the
function returns whether it wrote anything, so the reported count is of fields
that actually reached the file rather than rows the loop looked at. The empty
array the writer genuinely cannot encode is the one deliberate skip and now
reports itself as one.

Checked against every model here before making it strict: no field on any of
the 20 GGUF-sourced databases raises or falls through, so nothing that works
today starts failing. The 28 fields stored as `json` come from safetensors
imports, and `sqlite_to_gguf` rejects those databases before this loop.

The viewer's two catches keep the reason instead of flattening it to `True`.
A viewer over a few hundred tensors should not fail to render because one is
odd, but a systematic decoding fault should read as the same message on every
tensor rather than as a blank panel that looks like a property of the model.

The remaining twelve `except Exception` are probes of optional runtimes --
whether mlx has a GPU, whether cupy sees a device, whether a framework tensor
has `.float()` -- where any failure does mean "not available". Those were left
as they are.

### The redundant index

`idx_tensors_name` duplicated the index `UNIQUE (name)` already creates. Every
query plan that used it is served identically by the implicit one, checked
across lookup, ordered scan and covering-scan shapes.

Removed for clarity, not for speed. **Measured, it cost nothing**: 0.07 MB and
no time difference across three runs on a 2.5 GB model, because a model has a
few hundred tensor names -- the largest here has 678, whose index is about
0.03 MB. Databases written earlier still carry it and are unaffected.

`idx_tensors_dtype` stays. `reminis info` groups by dtype, and without it that
is a scan of a table whose rows are multi-megabyte blobs.

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
