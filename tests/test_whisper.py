"""Verify that a Whisper model runs correctly out of a reminis database.

A speech model can fail in a way a text model cannot: it produces a fluent
English sentence that is not what the audio said. Nothing about the output
looks wrong, so every check here compares against an independent reference
rather than against reminis itself.

  * the log-Mel features against ``transformers``'s own feature extractor
  * the encoder's hidden states against ``transformers`` in float32
  * the decoder's logits, and the order of its top-5, against the same
  * the greedy transcription, token for token

The pieces that have no external reference are checked against properties
instead: LayerNorm against its definition, the convolution against a direct
sum, the two backends against each other.

Skips rather than fails when the model or the reference is absent.
"""

import sqlite3
from pathlib import Path

import numpy as np

from reminis import audio as ra
from reminis.whisper import Whisper, decode_tokens, is_whisper

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DB = MODELS_DIR / "whisper-tiny.db"
CHECKPOINT = MODELS_DIR / "whisper-tiny"
# Kokoro ships sample speech, which is real audio rather than a tone.
SPEECH = MODELS_DIR / "kokoro-82m" / "samples" / "HEARME.wav"

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"  --  {detail}" if detail else ""))
        FAILURES.append(label)


def have_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def have_mlx():
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except Exception:
        return False


def sample_audio():
    """Real speech if it is here, otherwise a deterministic synthetic clip."""
    if SPEECH.exists():
        return ra.read_wav(SPEECH)[0]
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, 64000, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 220 * t)
            + 0.05 * rng.standard_normal(64000).astype(np.float32))


# -- audio front end ------------------------------------------------------


def test_filterbank_and_features():
    print("\nLog-Mel features against transformers")
    if not have_torch():
        print("  skip  transformers is not installed")
        return
    if not CHECKPOINT.exists():
        print(f"  skip  {CHECKPOINT} is not here")
        return
    from transformers import WhisperFeatureExtractor
    from transformers.audio_utils import mel_filter_bank

    reference = mel_filter_bank(
        num_frequency_bins=ra.N_FFT // 2 + 1, num_mel_filters=ra.N_MELS,
        min_frequency=0.0, max_frequency=ra.SAMPLE_RATE / 2,
        sampling_rate=ra.SAMPLE_RATE, norm="slaney", mel_scale="slaney").T
    worst = float(np.abs(ra.mel_filterbank() - reference).max())
    check(f"the Mel filterbank matches ({worst:.1e})", worst < 1e-6)

    extractor = WhisperFeatureExtractor.from_pretrained(str(CHECKPOINT))
    for label, clip in (
        ("silence", np.zeros(ra.SAMPLE_RATE * 3, dtype=np.float32)),
        ("speech", sample_audio()),
    ):
        want = extractor(clip, sampling_rate=ra.SAMPLE_RATE,
                         return_tensors="np").input_features[0]
        got = ra.log_mel(clip)
        gap = float(np.abs(got - want).max())
        check(f"features match on {label} ({gap:.1e})",
              got.shape == want.shape and gap < 1e-4,
              f"shape {got.shape} vs {want.shape}, worst {gap:.3e}")


def test_thirty_second_window():
    print("\nThe fixed 30-second window")
    short = ra.log_mel(np.zeros(ra.SAMPLE_RATE, dtype=np.float32))
    long = ra.log_mel(np.zeros(ra.SAMPLE_RATE * 90, dtype=np.float32))
    check("a short clip is padded to 3000 frames", short.shape == (80, 3000))
    check("a long clip is truncated to 3000 frames", long.shape == (80, 3000))


def test_wav_reading():
    print("\nReading wav files")
    if not SPEECH.exists():
        print(f"  skip  {SPEECH} is not here")
        return
    clip, rate = ra.read_wav(SPEECH)
    check("mono float32 comes back", clip.ndim == 1 and clip.dtype == np.float32)
    check("the source rate is reported", rate > 0)
    check("samples are inside [-1, 1]", float(np.abs(clip).max()) <= 1.0)
    # Resampling must preserve duration, which is the thing a wrong ratio
    # silently changes -- and a clip at the wrong speed still transcribes to
    # confident English.
    seconds = len(clip) / ra.SAMPLE_RATE
    with __import__("wave").open(str(SPEECH), "rb") as f:
        original = f.getnframes() / f.getframerate()
    check(f"duration survives resampling ({seconds:.2f}s vs {original:.2f}s)",
          abs(seconds - original) < 0.05)


# -- the forward pass in pieces -------------------------------------------


def test_layer_norm_and_conv():
    print("\nThe two operations reminis had no implementation of")
    from reminis.backend import select as select_backend
    from reminis.whisper import _conv1d, _layer_norm

    backend = select_backend("inference", "numpy")
    xp = backend.xp
    rng = np.random.default_rng(1)

    x = rng.standard_normal((7, 16)).astype(np.float32) * 3.0
    weight = rng.standard_normal(16).astype(np.float32)
    bias = rng.standard_normal(16).astype(np.float32)
    got = np.asarray(_layer_norm(backend, x, weight, bias, 1e-5))
    mu = x.mean(-1, keepdims=True)
    sd = np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5)
    want = (x - mu) / sd * weight + bias
    check("LayerNorm matches its definition",
          float(np.abs(got - want).max()) < 1e-5)
    # RMSNorm skips the centring; if the two agreed, the test would be
    # passing for the wrong reason.
    rms = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * weight + bias
    check("and is not RMSNorm", float(np.abs(got - rms).max()) > 1e-3)

    # MLX substitutes a fused kernel for the eleven-operation form. It has to
    # agree with the definition too, or the fast path is a different model.
    if have_mlx():
        import mlx.core as mx

        fast = select_backend("inference", "mlx")
        out = fast.layer_norm(mx.array(x), mx.array(weight), mx.array(bias), 1e-5)
        gap = float(np.abs(np.asarray(out, dtype=np.float32) - want).max())
        check(f"the fused kernel agrees with the definition ({gap:.1e})",
              gap < 1e-4, f"worst {gap:.3e}")

    signal = rng.standard_normal((5, 24)).astype(np.float32)
    kernel = rng.standard_normal((6, 5, 3)).astype(np.float32)
    kbias = rng.standard_normal(6).astype(np.float32)
    for stride in (1, 2):
        got = np.asarray(_conv1d(xp, signal, kernel, kbias, stride))
        padded = np.pad(signal, ((0, 0), (1, 1)))
        n_out = 1 + (padded.shape[1] - 3) // stride
        want = np.empty((6, n_out), dtype=np.float32)
        for o in range(6):
            for t in range(n_out):
                window = padded[:, t * stride:t * stride + 3]
                want[o, t] = (kernel[o] * window).sum() + kbias[o]
        check(f"conv1d matches a direct sum at stride {stride}",
              got.shape == want.shape
              and float(np.abs(got - want).max()) < 1e-4)


# -- against the reference implementation ---------------------------------


def _reference(mel, primed):
    import torch
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        str(CHECKPOINT), torch_dtype=torch.float32).eval()
    features = torch.from_numpy(mel)[None]
    with torch.no_grad():
        encoded = model.get_encoder()(features).last_hidden_state[0].numpy()
        logits = model(input_features=features,
                       decoder_input_ids=torch.tensor([primed])).logits[0].numpy()
        generated = model.generate(features, max_new_tokens=224)[0].tolist()
    return encoded, logits, generated


def test_against_transformers():
    print("\nEncoder, decoder and transcription against transformers")
    if not have_torch():
        print("  skip  transformers is not installed")
        return
    if not (DB.exists() and CHECKPOINT.exists()):
        print(f"  skip  {DB} or {CHECKPOINT} is not here")
        return
    from reminis.backend import select as select_backend

    mel = ra.log_mel(sample_audio())
    model = Whisper(str(DB), backend=select_backend("inference", "numpy"))
    primed = model.prompt_tokens()
    want_enc, want_logits, want_ids = _reference(mel, primed)

    encoded = model.encode(mel)
    model.backend.eval(encoded)
    got_enc = np.asarray(model.backend.to_numpy(encoded), dtype=np.float32)
    rel = float(np.abs(got_enc - want_enc).max() / np.abs(want_enc).max())
    corr = float(np.corrcoef(got_enc.ravel(), want_enc.ravel())[0, 1])
    check(f"encoder hidden states agree (rel {rel:.1e}, corr {corr:.8f})",
          rel < 1e-3 and corr > 0.99999)

    ekv = model.encoder_kv(encoded)
    caches = [None] * model.cfg.n_decoder_layers
    got_logits = model.decode(primed, ekv, caches, 0)
    rel = float(np.abs(got_logits - want_logits).max()
                / np.abs(want_logits).max())
    corr = float(np.corrcoef(got_logits.ravel(), want_logits.ravel())[0, 1])
    check(f"decoder logits agree (rel {rel:.1e}, corr {corr:.8f})",
          rel < 1e-3 and corr > 0.99999)

    got_top = np.argsort(got_logits[-1])[::-1][:5].tolist()
    want_top = np.argsort(want_logits[-1])[::-1][:5].tolist()
    check("the top-5 next tokens are identical, in order", got_top == want_top,
          f"{got_top} vs {want_top}")

    got_ids = model.transcribe(mel)
    want_produced = [t for t in want_ids if t not in primed and t != model.cfg.eos]
    check(f"the greedy transcription matches token for token "
          f"({len(got_ids)} tokens)",
          got_ids == want_produced,
          f"{got_ids[:12]} vs {want_produced[:12]}")
    model.close()


def test_backends_agree():
    print("\nThe backends against each other")
    if not DB.exists():
        print(f"  skip  {DB} is not here")
        return
    if not have_mlx():
        print("  skip  mlx is not available")
        return
    from reminis.backend import select as select_backend

    mel = ra.log_mel(sample_audio())
    texts = {}
    for name in ("numpy", "mlx"):
        model = Whisper(str(DB), backend=select_backend("inference", name))
        texts[name] = model.transcribe(mel)
        model.close()
    check("mlx and numpy produce the same tokens",
          texts["numpy"] == texts["mlx"],
          f"{texts['numpy'][:12]} vs {texts['mlx'][:12]}")


# -- the database as a complete model -------------------------------------


def test_both_whisper_families():
    """The English-only checkpoints number every special token one lower.

    whisper-tiny starts the decoder at 50258 and marks no-timestamps with
    50363; whisper-tiny.en uses 50257 and 50362, because it carries no
    language or task tokens. Hardcoding either set primes the other model's
    decoder with the wrong prefix -- which transcribes fluently and wrongly
    rather than failing, so this compares the whole transcription against
    the reference for each family that is present.
    """
    print("\nMultilingual and English-only checkpoints")
    if not have_torch():
        print("  skip  transformers is not installed")
        return
    import torch
    from transformers import WhisperForConditionalGeneration

    mel = ra.log_mel(sample_audio())
    families = [
        ("whisper-tiny", True, 4),
        ("whisper-tiny-en", False, 4),
        ("whisper-base", True, 6),
    ]
    seen = 0
    for name, multilingual, layers in families:
        db = MODELS_DIR / f"{name}.db"
        checkpoint = MODELS_DIR / name
        if not (db.exists() and checkpoint.exists()):
            continue
        seen += 1
        model = Whisper(str(db))
        primed = model.prompt_tokens()
        check(f"{name}: multilingual is {multilingual}",
              model.cfg.multilingual is multilingual)
        check(f"{name}: {layers} encoder layers were read",
              model.cfg.n_encoder_layers == layers)
        # The prefix must start where the config says, not where a constant
        # for the other family says.
        check(f"{name}: primed at {primed[0]}", primed[0] == model.cfg.start)
        check(f"{name}: no id repeats in the prefix", len(set(primed)) == len(primed))

        reference = WhisperForConditionalGeneration.from_pretrained(
            str(checkpoint), torch_dtype=torch.float32).eval()
        with torch.no_grad():
            want = reference.generate(torch.from_numpy(mel)[None],
                                      max_new_tokens=224)[0].tolist()
        want_produced = [t for t in want
                         if t not in primed and t != model.cfg.eos]
        got = model.transcribe(mel)
        check(f"{name}: transcription matches transformers ({len(got)} tokens)",
              got == want_produced,
              f"{got[:10]} vs {want_produced[:10]}")
        model.close()

    if seen == 0:
        print("  skip  no whisper checkpoints are here")
    elif seen == 1:
        print("  note  only one family present; the other was not checked")


def test_tokenizer_is_in_the_database():
    print("\nThe tokenizer travels with the weights")
    if not DB.exists():
        print(f"  skip  {DB} is not here")
        return
    conn = sqlite3.connect(DB)
    try:
        meta = dict(conn.execute("SELECT key, value FROM model_meta"))
    finally:
        conn.close()

    check("the database says it is a Whisper model", is_whisper(meta))
    check("a vocabulary is stored", bool(meta.get("tokenizer.ggml.tokens")))

    from reminis.infer import _parse_array

    tokens = _parse_array(meta, "tokenizer.ggml.tokens")
    vocab_size = int(meta.get("config.vocab_size", 0))
    check(f"the vocabulary is the size the config claims ({len(tokens)})",
          len(tokens) == vocab_size, f"{len(tokens)} vs {vocab_size}")
    for name, index in (("<|startoftranscript|>", 50258),
                        ("<|endoftext|>", 50257),
                        ("<|transcribe|>", 50359),
                        ("<|notimestamps|>", 50363)):
        check(f"{name} is at {index}",
              index < len(tokens) and tokens[index] == name,
              f"found {tokens[index]!r}" if index < len(tokens) else "missing")

    # Decoding must go through reminis's own tokenizer, not transformers'.
    text = decode_tokens(str(DB), [50364, 2425, 11, 1002, 13])
    check("ids decode to text with no help from transformers",
          isinstance(text, str) and len(text) > 0, repr(text))


def test_end_to_end_transcription():
    print("\nEnd to end, out of the database alone")
    if not (DB.exists() and SPEECH.exists()):
        print("  skip  the model or the sample audio is not here")
        return
    from reminis.whisper import transcribe_file

    result = transcribe_file(str(DB), str(SPEECH), verbose=False)
    text = (result["text"] or "").lower()
    check("it produced text", bool(text.strip()), repr(text))
    # The clip is Kokoro's own description of itself, so these words are in
    # it. A wrong forward pass gives fluent English that is not this.
    for word in ("open", "model", "parameters", "quality"):
        check(f"the transcription contains {word!r}", word in text)
    check("the source rate was reported", result["source_rate"] > 0)
    check("nothing was truncated for a 21-second clip",
          result["truncated"] is False)


def test_refusals():
    print("\nWhat it refuses")
    from reminis.whisper import UnsupportedModel

    text_model = MODELS_DIR / "SmolLM-135M.f16.db"
    if text_model.exists():
        try:
            Whisper(str(text_model))
            check("a text model is refused", False, "it was accepted")
        except UnsupportedModel as exc:
            check("a text model is refused, by name",
                  "Whisper" in str(exc) and "reminis run" in str(exc))
    else:
        print("  skip  no text model to try")

    check("is_whisper says no to a llama model",
          not is_whisper({"general.architecture": "llama"}))
    check("is_whisper reads model_type as well as architecture",
          is_whisper({"config.model_type": '"whisper"'}))

    if not have_torch():
        return
    # A 24-bit wav is not PCM reminis reads; it must say so rather than
    # interpret the bytes as 16-bit and transcribe noise.
    import wave

    tmp = ROOT / "tests" / "tmp"
    tmp.mkdir(exist_ok=True)
    odd = tmp / "24bit.wav"
    with wave.open(str(odd), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(3)
        f.setframerate(16000)
        f.writeframes(b"\x00" * 3 * 1000)
    try:
        ra.read_wav(odd)
        check("a 24-bit wav is refused", False, "it was read anyway")
    except ValueError as exc:
        check("a 24-bit wav is refused, with a way out", "ffmpeg" in str(exc))
    finally:
        odd.unlink(missing_ok=True)
