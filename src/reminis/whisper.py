"""Run Whisper out of a reminis database.

``infer.py`` runs decoder-only text models: one embedding table, one stack of
blocks, RMSNorm, SwiGLU, rotary positions, a causal cache. Whisper shares
almost none of that. It is an encoder-decoder; it normalises with LayerNorm
and a bias; its feed-forward is GELU with no gate; its positions are a stored
table rather than a rotation; its encoder begins with two 1-D convolutions
over a spectrogram; and every decoder block attends twice -- once to the text
so far and once to the audio.

Expressing that as an entry in ``arch.py`` would be a fiction. That registry
holds *deviations* from a shared block, and here there is no shared block
left. So this is its own forward pass, over the same ``WeightStore`` and the
same backends.

Verified against ``transformers`` running the original checkpoint in float32
on identical features:

    encoder hidden states     correlation 1.00000000, 1.0e-04 relative
    decoder logits            correlation 1.00000000, 2.0e-06 relative
    top-5 next tokens         identical, in order
    greedy transcription      token-for-token identical

On MLX the weights are float16, so the same comparison is 6e-02 relative on
the encoder and 8e-04 on the logits -- and the transcription is still
identical, token for token, to both numpy and the reference.

The point is the one ``reminis run`` makes for text. A database that can
transcribe is a complete model rather than an archive of one -- so a speech
model that has been merged, rolled back, or rebuilt from a delta pack can be
checked by asking it to listen.
"""

import json
import os
import sqlite3

import numpy as np

from reminis.audio import N_MELS, log_mel, read_wav

# The tokens Whisper's decoder is primed with, for the multilingual
# checkpoints. A transcription is a completion of this prefix, so getting it
# wrong does not fail -- it translates when you asked it to transcribe, or
# writes timestamps you did not want.
#
# These are fallbacks only. **The English-only checkpoints shift every one of
# them down by one**, because they drop the 99 language tokens and the task
# tokens: whisper-tiny.en starts at 50257 and marks no-timestamps with 50362,
# where the multilingual model uses 50258 and 50363. Hardcoding either set
# gives the other one a decoder primed with the wrong prefix, which produces
# fluent English that is not what was said. So they are looked up by name in
# the stored vocabulary first, and only fall back to these.
START_OF_TRANSCRIPT = 50258
END_OF_TEXT = 50257
TRANSLATE = 50358
TRANSCRIBE = 50359
NO_TIMESTAMPS = 50363
# <|en|> heads the 99 language tokens, which run in a fixed order.
LANG_EN = 50259


class UnsupportedModel(Exception):
    pass


def is_whisper(meta: dict) -> bool:
    """Whether this database holds a Whisper model."""
    if str(meta.get("config.model_type", "")).strip('"').lower() == "whisper":
        return True
    return "whisper" in str(meta.get("general.architecture", "")).lower()


class _Config:
    """Whisper's hyperparameters, out of the config the conversion kept."""

    def __init__(self, meta: dict):
        def value(key, default=None):
            raw = meta.get(f"config.{key}")
            if raw is None:
                if default is None:
                    raise UnsupportedModel(
                        f"This database has no config.{key}, which Whisper "
                        "needs. Was it converted from a Whisper checkpoint?"
                    )
                return default
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
            return raw

        self.d_model = int(value("d_model"))
        self.n_encoder_layers = int(value("encoder_layers"))
        self.n_decoder_layers = int(value("decoder_layers"))
        self.n_encoder_heads = int(value("encoder_attention_heads"))
        self.n_decoder_heads = int(value("decoder_attention_heads"))
        self.n_mels = int(value("num_mel_bins", N_MELS))
        self.vocab_size = int(value("vocab_size"))
        self.max_target = int(value("max_target_positions", 448))
        self.eos = int(value("eos_token_id", END_OF_TEXT))
        self.suppress = [int(t) for t in value("suppress_tokens", [])]
        self.begin_suppress = [int(t) for t in value("begin_suppress_tokens", [])]
        # The English-only checkpoints have no language or task tokens, so
        # their vocabulary is one short of the multilingual one.
        self.multilingual = self.vocab_size > 51864
        # Whisper records where the decoder starts, which is the one special
        # token whose id differs between the two families and is written down.
        self.start = int(value("decoder_start_token_id", START_OF_TRANSCRIPT))
        # LayerNorm's epsilon is not recorded in Whisper's config; it is
        # torch's default, which is what the checkpoint was trained under.
        self.eps = 1e-5


class _Weights:
    """Weights by name, fetched once and kept.

    ``WeightStore`` already caches, but a bias wants flattening and every
    lookup here happens inside the decode loop, so the shaped array is
    remembered rather than reshaped per token.
    """

    def __init__(self, store, backend):
        self.store = store
        self.backend = backend
        self._cache = {}

    def __call__(self, name):
        hit = self._cache.get(name)
        if hit is None:
            hit = self.store.get(name)
            if name.endswith(".bias"):
                hit = hit.reshape(-1)
            self._cache[name] = hit
        return hit


def _erf(xp, x):
    """The error function, exactly where the backend has one.

    MLX ships ``erf``; numpy does not, and the tanh approximation everyone
    reaches for instead is off by about 1e-3 -- large enough to show up
    against the reference. Abramowitz & Stegun 7.1.26 is within 1.5e-7,
    below float32 epsilon, so the two backends agree to float noise rather
    than to the width of an approximation.
    """
    fn = getattr(xp, "erf", None)
    if fn is not None:
        return fn(x)
    sign = xp.sign(x)
    x = xp.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - poly * xp.exp(-x * x))


def _gelu(xp, x):
    """The exact GELU, which is what Whisper was trained with."""
    return 0.5 * x * (1.0 + _erf(xp, x * 0.7071067811865476))


def _layer_norm(backend, x, weight, bias, eps):
    """LayerNorm: centred as well as scaled, and with a bias.

    Not RMSNorm. Dropping the mean subtraction leaves a model that still
    produces words, which is exactly why it is worth naming.

    Delegated to the backend, which uses a fused kernel where it has one.
    Written out it is eleven operations, and with three norms per decoder
    layer that was two thirds of everything dispatched for a token -- the
    dominant cost on a model this small. See `Backend.layer_norm` for why the
    statistics are taken in float32 whatever the weights are stored as.
    """
    return backend.layer_norm(x, weight, bias, eps)


def _conv1d(xp, x, weight, bias, stride):
    """1-D convolution as one matrix product per kernel tap.

    `x` is (in_channels, time) and `weight` is (out, in, k), which is the
    layout the checkpoint stores. Whisper's kernels are 3 wide, so this is
    three matrix products and two adds -- cheaper to write and to run than
    building an im2col matrix, and it needs no fancy indexing, which the
    backends spell differently.
    """
    out_channels, _, k = weight.shape
    pad = k // 2
    padded = xp.pad(x, ((0, 0), (pad, pad)))
    n_out = 1 + (padded.shape[1] - k) // stride
    span = (n_out - 1) * stride + 1

    acc = None
    for tap in range(k):
        window = padded[:, tap:tap + span:stride]
        term = weight[:, :, tap] @ window
        acc = term if acc is None else acc + term
    return acc + bias.reshape(-1, 1)


def _split_heads(x, n_heads):
    """(tokens, d_model) -> (1, heads, tokens, head_dim)."""
    tokens, d = x.shape
    return x.reshape(tokens, n_heads, d // n_heads).transpose(1, 0, 2)[None]


def _merge_heads(x, tokens, d_model):
    """(1, heads, tokens, head_dim) -> (tokens, d_model)."""
    return x[0].transpose(1, 0, 2).reshape(tokens, d_model)


class Whisper:
    """Whisper's forward pass over weights selected out of SQLite."""

    def __init__(self, db_path, backend=None):
        from reminis.backend import select as select_backend
        from reminis.infer import WeightStore

        self.backend = backend or select_backend("inference")
        self.store = WeightStore(db_path, backend=self.backend)
        meta = dict(self.store.conn.execute("SELECT key, value FROM model_meta"))
        if not is_whisper(meta):
            raise UnsupportedModel(
                f"{db_path} does not hold a Whisper model. "
                "`reminis transcribe` reads speech models; use `reminis run` "
                "for a text model."
            )
        self.meta = meta
        self.cfg = _Config(meta)
        self.w = _Weights(self.store, self.backend)
        self._tokenizer = None
        self._suppress = None
        self._begin_suppress = None
        self._vocab_cache = None
        self._prompt_cache = {}

    @property
    def _vocab(self):
        """The token list, parsed once. Empty when the database has none."""
        if self._vocab_cache is None:
            from reminis.infer import _parse_array

            self._vocab_cache = _parse_array(self.meta, "tokenizer.ggml.tokens")
        return self._vocab_cache

    # -- encoder ----------------------------------------------------------

    def encode(self, mel):
        """The audio's hidden states: (frames, d_model).

        Two convolutions with a GELU after each -- the second strided, which
        is what halves 3000 spectrogram frames into the 1500 positions the
        rest of the model is built around -- then a stored positional table
        and the ordinary pre-norm blocks.
        """
        cfg, w, b = self.cfg, self.w, self.backend
        xp = b.xp

        h = b.from_numpy(np.ascontiguousarray(mel))
        h = _gelu(xp, _conv1d(xp, h, w("model.encoder.conv1.weight"),
                              w("model.encoder.conv1.bias"), stride=1))
        h = _gelu(xp, _conv1d(xp, h, w("model.encoder.conv2.weight"),
                              w("model.encoder.conv2.bias"), stride=2))
        h = h.T

        # Whisper's encoder positions are sinusoids, but they are *stored*
        # rather than derived, so there is nothing to recompute -- and a
        # shorter clip simply uses the front of the table.
        positions = w("model.encoder.embed_positions.weight")
        h = h + positions[:h.shape[0]]

        for layer in range(cfg.n_encoder_layers):
            p = f"model.encoder.layers.{layer}."
            residual = h
            h = _layer_norm(b, h, w(p + "self_attn_layer_norm.weight"),
                            w(p + "self_attn_layer_norm.bias"), cfg.eps)
            h = residual + self._attend(h, p + "self_attn.", cfg.n_encoder_heads)
            h = self._feed_forward(h, p)

        return _layer_norm(b, h, w("model.encoder.layer_norm.weight"),
                           w("model.encoder.layer_norm.bias"), cfg.eps)

    # -- shared pieces ----------------------------------------------------

    def _project_kv(self, x, prefix, n_heads):
        """Keys and values. The key projection carries no bias, by design.

        Whisper omits it because a bias on the keys cancels in the softmax;
        reading its absence as a missing weight rather than as a deliberate
        omission is the obvious way to get this wrong.

        **Stacking these into one matrix product was tried and lost.** Fusing
        Q, K and V is worth a third in the text path, where a token is a
        matrix-vector product against a large matrix. Here the projections
        are 384x384 and a decoded token is one row, so the products are
        dispatch-bound rather than arithmetic-bound -- and the three slices
        and the reshape-and-transpose of each strided view cost more than the
        two dispatches saved. Measured: 1.84 ms a token became 3.14 ms, and
        the whole transcription went from 137 ms to 220 ms.
        """
        w = self.w
        k = x @ w(prefix + "k_proj.weight").T
        v = x @ w(prefix + "v_proj.weight").T + w(prefix + "v_proj.bias")
        return _split_heads(k, n_heads), _split_heads(v, n_heads)

    def _attend(self, x, prefix, n_heads, keys=None, values=None, mask=None,
                query=None):
        """One attention sub-block, self or cross.

        Cross-attention is the same computation with someone else's keys and
        values, so it is the same code with them passed in.
        """
        cfg, w, b = self.cfg, self.w, self.backend
        tokens = x.shape[0]
        head_dim = cfg.d_model // n_heads

        if query is None:
            query = _split_heads(x @ w(prefix + "q_proj.weight").T
                                 + w(prefix + "q_proj.bias"), n_heads)
        if keys is None:
            keys, values = self._project_kv(x, prefix, n_heads)

        out = b.attention(query, keys, values, head_dim ** -0.5, mask)
        out = _merge_heads(out, tokens, cfg.d_model)
        return out @ w(prefix + "out_proj.weight").T + w(prefix + "out_proj.bias")

    def _feed_forward(self, h, p):
        """One GELU between two matrices -- no gate, so no third matrix.

        SwiGLU, which every text model here uses, needs a gate projection
        alongside the up projection and multiplies the two. Whisper does not.
        """
        cfg, w, b = self.cfg, self.w, self.backend
        xp = b.xp
        residual = h
        h = _layer_norm(b, h, w(p + "final_layer_norm.weight"),
                        w(p + "final_layer_norm.bias"), cfg.eps)
        h = _gelu(xp, h @ w(p + "fc1.weight").T + w(p + "fc1.bias"))
        return residual + (h @ w(p + "fc2.weight").T + w(p + "fc2.bias"))

    # -- decoder ----------------------------------------------------------

    def encoder_kv(self, encoded):
        """Cross-attention keys and values for every decoder layer.

        These depend on the audio alone, so they are computed once for the
        whole transcription rather than per token. On a 1500-frame encoding
        that is the difference between four projections and four thousand.
        """
        out = []
        for layer in range(self.cfg.n_decoder_layers):
            p = f"model.decoder.layers.{layer}.encoder_attn."
            out.append(self._project_kv(encoded, p, self.cfg.n_decoder_heads))
        return out

    def decode(self, tokens, encoder_kv, caches, offset):
        """Logits for `tokens` as numpy, for comparing against a reference.

        Generation uses `_decode` and never brings the whole vocabulary back
        from the device; this is the path that wants every position in a form
        numpy can diff.
        """
        logits = self._decode(tokens, encoder_kv, caches, offset,
                              last_only=False)
        self.backend.eval(logits)
        return np.asarray(self.backend.to_numpy(logits), dtype=np.float32)

    def _decode(self, tokens, encoder_kv, caches, offset, last_only=True):
        """Logits on the device, given the audio and the text so far.

        `last_only` projects just the final position through the vocabulary.
        Decoding never needs the others, and that projection is the single
        largest operation in a token -- it reads the whole 51,865-row
        embedding table -- so doing it four times during prefill and throwing
        three away is pure waste.
        """
        cfg, w, b = self.cfg, self.w, self.backend
        xp = b.xp
        n = len(tokens)

        ids = xp.array(np.asarray(tokens, dtype=np.int64))
        h = xp.take(w("model.decoder.embed_tokens.weight"), ids, axis=0)
        h = h + w("model.decoder.embed_positions.weight")[offset:offset + n]

        mask = None
        if n > 1:
            # Only the prefill sees more than one token, and only it needs a
            # causal mask; a single token attends to everything cached and
            # there is nothing to hide.
            #
            # The backends take a boolean *keep* mask -- true where a query
            # may see a key -- not an additive one. Handing them additive
            # -inf inverts it, every row masks itself out, and the softmax
            # returns NaN.
            rows = np.arange(offset, offset + n)[:, None]
            cols = np.arange(offset + n)[None, :]
            mask = xp.array(cols <= rows)

        for layer in range(cfg.n_decoder_layers):
            p = f"model.decoder.layers.{layer}."
            residual = h
            h = _layer_norm(b, h, w(p + "self_attn_layer_norm.weight"),
                            w(p + "self_attn_layer_norm.bias"), cfg.eps)
            keys, values = self._project_kv(h, p + "self_attn.",
                                            cfg.n_decoder_heads)
            cached = caches[layer]
            if cached is not None:
                keys = xp.concatenate([cached[0], keys], axis=2)
                values = xp.concatenate([cached[1], values], axis=2)
            caches[layer] = (keys, values)
            h = residual + self._attend(h, p + "self_attn.", cfg.n_decoder_heads,
                                        keys, values, mask)

            residual = h
            h = _layer_norm(b, h, w(p + "encoder_attn_layer_norm.weight"),
                            w(p + "encoder_attn_layer_norm.bias"), cfg.eps)
            ekv = encoder_kv[layer]
            h = residual + self._attend(h, p + "encoder_attn.",
                                        cfg.n_decoder_heads, ekv[0], ekv[1])

            h = self._feed_forward(h, p)

        h = _layer_norm(b, h, w("model.decoder.layer_norm.weight"),
                        w("model.decoder.layer_norm.bias"), cfg.eps)
        if last_only:
            h = h[-1:]
        # Whisper ties its output projection to the embedding table.
        return h @ w("model.decoder.embed_tokens.weight").T

    # -- transcription ----------------------------------------------------

    def prompt_tokens(self, language="en", task="transcribe", timestamps=False):
        """The prefix a transcription completes.

        Every id here is looked up by name in the vocabulary the database
        carries, because the English-only checkpoints number them one lower
        than the multilingual ones and a wrong prefix transcribes fluently
        and wrongly rather than failing.
        """
        key = (language, task, timestamps)
        hit = self._prompt_cache.get(key)
        if hit is not None:
            return list(hit)

        vocab = self._vocab
        ids = [self.cfg.start]
        if self.cfg.multilingual:
            # `<|en|>` heads the language block on every multilingual
            # checkpoint, so it is a safe fallback for a database converted
            # before conversions carried their tokenizer. No other language
            # is, since its id depends on where in the 99 the code sits.
            ids.append(_special(vocab, f"<|{language}|>",
                                LANG_EN if language == "en" else None))
            marker = "<|transcribe|>" if task == "transcribe" else "<|translate|>"
            ids.append(_special(vocab, marker,
                                TRANSCRIBE if task == "transcribe" else TRANSLATE))
        if not timestamps:
            ids.append(_special(vocab, "<|notimestamps|>", NO_TIMESTAMPS))
        self._prompt_cache[key] = tuple(ids)
        return ids

    def _suppression_masks(self):
        """The two suppression lists as additive device arrays, built once.

        Whisper's config lists tokens that must never be produced -- the
        non-speech markers and the specials -- and a second list barred only
        from the first position. Skipping them gives fluent output that
        quietly disagrees with every other implementation; it was the last
        thing still wrong here once the logits already matched to 2e-06.

        They are held as arrays rather than index lists so the masking is one
        addition on the device instead of a scatter on the host, which is
        what lets the whole sampling step stay off the host.
        """
        if self._suppress is None:
            xp = self.backend.xp
            vocab = self.cfg.vocab_size

            def additive(indices):
                if not indices:
                    return None
                block = np.zeros(vocab, dtype=np.float32)
                block[[i for i in indices if 0 <= i < vocab]] = -np.inf
                return xp.array(block)

            self._suppress = (additive(self.cfg.suppress),)
            self._begin_suppress = (additive(self.cfg.begin_suppress),)
        return self._suppress[0], self._begin_suppress[0]

    def transcribe(self, mel, max_tokens=224, language="en", task="transcribe",
                   temperature=0.0, seed=None, on_token=None):
        """Greedy (or sampled) decoding of one 30-second window.

        Greedy decoding never leaves the device except to carry one integer
        back per token. Bringing the logits home instead means copying the
        whole 51,865-wide row and stalling the pipeline every step -- 0.23 ms
        a token, which on a model this small is a tenth of the token.
        """
        cfg, b = self.cfg, self.backend
        xp = b.xp
        encoded = self.encode(mel)
        b.eval(encoded)
        ekv = self.encoder_kv(encoded)
        caches = [None] * cfg.n_decoder_layers
        always, at_begin = self._suppression_masks()

        primed = self.prompt_tokens(language, task)
        logits = self._decode(primed, ekv, caches, 0)[-1]

        rng = np.random.default_rng(seed)
        produced = []
        for step in range(max_tokens):
            if always is not None:
                logits = logits + always
            if step == 0 and at_begin is not None:
                logits = logits + at_begin

            if temperature <= 0:
                # argmax on the device, then one scalar across the boundary.
                nxt = int(xp.argmax(logits).item())
            else:
                host = np.asarray(b.to_numpy(logits), dtype=np.float32)
                scaled = host / np.float32(temperature)
                scaled -= scaled.max()
                probs = np.exp(scaled)
                probs /= probs.sum()
                nxt = int(rng.choice(len(probs), p=probs))

            if nxt == cfg.eos:
                break
            produced.append(nxt)
            if on_token is not None:
                on_token(nxt)
            if len(primed) + len(produced) >= cfg.max_target:
                break
            logits = self._decode([nxt], ekv, caches,
                                  len(primed) + len(produced) - 1)[-1]

        return produced

    def detokenize(self, ids):
        """Token ids as text, with the tokenizer built once and kept.

        Rebuilding it per call costs 262 ms on a 51,865-token vocabulary --
        the merge list has to be read and ranked -- against 0.04 ms to
        actually decode. That was 40% of a transcription before it was
        cached, and none of it was arithmetic.
        """
        if self._tokenizer is None:
            from reminis.infer import build_tokenizer

            if not self.meta.get("tokenizer.ggml.tokens"):
                raise UnsupportedModel(
                    "This database has no tokenizer in it, so the ids cannot "
                    "be turned back into text. Re-convert the checkpoint from "
                    "a directory with its tokenizer.json beside the weights."
                )
            self._tokenizer = build_tokenizer(self.meta)
        return self._tokenizer.decode(list(ids))

    def close(self):
        self.store.close()


def _special(vocab, name, fallback):
    """A special token's id, by name, out of the vocabulary.

    Looking these up rather than hardcoding them is what makes the same code
    drive both Whisper families: they order the specials identically but at
    different offsets, so a name is stable where a number is not.

    `vocab` is the already-parsed token list, because parsing it is not
    cheap: it is stored as the repr of 51,865 strings, and reading that back
    costs 79 ms. Doing it once per lookup made priming the decoder -- three
    lookups -- cost 236 ms, which was 65% of a whole transcription and none
    of it arithmetic.

    `fallback` is used when the database carries no tokenizer. Passing None
    means there is no safe fallback -- a language token, whose id depends on
    where in the 99 the code sits -- so the caller is told rather than given
    a guess.
    """
    if vocab:
        try:
            return vocab.index(name)
        except ValueError:
            raise UnsupportedModel(
                f"This model's tokenizer has no {name}. "
                f"For a language, pass a code it knows; whisper-*.en "
                f"checkpoints are English-only and take no language."
            ) from None
    if fallback is not None:
        return fallback
    raise UnsupportedModel(
        f"Resolving {name} needs the tokenizer, which this database does not "
        "carry. Re-convert the checkpoint from a directory with its "
        "tokenizer.json beside the weights."
    )


_TOKENIZERS = {}


def decode_tokens(db_path, ids):
    """Token ids as text, using the tokenizer stored in the database.

    Memoised on the database's path and modification time, because building
    a 51,865-token BPE tokenizer costs 262 ms and decoding with it costs
    0.04 ms. Prefer ``Whisper.detokenize`` when a model is already open.
    """
    from reminis.infer import build_tokenizer

    try:
        stamp = os.stat(db_path).st_mtime_ns
    except OSError:
        stamp = None
    key = (str(db_path), stamp)

    tokenizer = _TOKENIZERS.get(key)
    if tokenizer is None:
        conn = sqlite3.connect(db_path)
        try:
            meta = dict(conn.execute("SELECT key, value FROM model_meta"))
        finally:
            conn.close()
        if not meta.get("tokenizer.ggml.tokens"):
            raise UnsupportedModel(
                "This database has no tokenizer in it, so the ids cannot be "
                "turned back into text. Re-convert the checkpoint from a "
                "directory that has its tokenizer.json beside the weights."
            )
        tokenizer = build_tokenizer(meta)
        _TOKENIZERS.clear()
        _TOKENIZERS[key] = tokenizer
    return tokenizer.decode(list(ids))


def transcribe_file(db_path, audio_path, max_tokens=224, language="en",
                    task="transcribe", temperature=0.0, seed=None,
                    backend=None, verbose=True):
    """Transcribe an audio file with a Whisper model held in a database.

    `verbose` is accepted so the signature matches ``generate``; the caller
    decides what to print, and this returns everything it would need.
    """
    import time

    from reminis.audio import SAMPLE_RATE

    audio, source_rate = read_wav(audio_path)
    seconds = len(audio) / SAMPLE_RATE
    mel = log_mel(audio)

    model = Whisper(db_path, backend=backend)
    try:
        started = time.time()
        ids = model.transcribe(mel, max_tokens=max_tokens, language=language,
                               task=task, temperature=temperature, seed=seed)
        elapsed = time.time() - started
        try:
            text = model.detokenize(ids)
        except UnsupportedModel:
            text = None
        return {
            "text": text,
            "tokens": ids,
            "seconds": seconds,
            "source_rate": source_rate,
            "elapsed": elapsed,
            "truncated": seconds > 30,
            "backend": model.backend.describe(),
            "layers": (model.cfg.n_encoder_layers, model.cfg.n_decoder_layers),
        }
    finally:
        model.close()
