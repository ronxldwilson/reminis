# Changelog

Measurements are on `Llama-3.2-1B-Instruct-f16` (2.5 GB, 147 tensors) unless
another model is named, taken on an M-series Mac with 17 GB of memory. Where a
release reports a before and an after, both were measured on the same machine
in the same cache state -- the earlier code was checked back out and re-run
rather than compared against a number from earlier in the day.

`bench.py` produces these figures; `tests/test_prerelease.py` is the gate every
release below had to pass.

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
