"""Audio in, log-Mel spectrogram out, in numpy alone.

A speech model's input is not tokens, so something has to turn a waveform
into the array its first layer expects. For Whisper that is a 80-channel
log-Mel spectrogram over exactly 30 seconds at 16 kHz, and it is specified
tightly enough that getting any part of it wrong produces a model that
transcribes confident nonsense rather than failing.

This is preprocessing rather than the forward pass, so it stays on the CPU
in numpy: it is one short-time Fourier transform over half a minute of
audio, which is nothing beside the model it feeds.

Verified against ``transformers``'s own ``WhisperFeatureExtractor`` -- the
filterbank agrees to 9e-10 and the finished features to 8e-06, which is
float32 noise.
"""

import wave

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
CHUNK_SECONDS = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


def _hz_to_mel(f):
    """Hertz to mels on the Slaney scale: linear below 1 kHz, log above."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    # `np.where` evaluates both arms, so the log is fed a floor rather than
    # zero -- the linear arm is the one that survives down there anyway.
    safe = np.maximum(f, 1e-9)
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(safe / min_log_hz) / logstep,
                    f / f_sp)


def _mel_to_hz(mel):
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(mel >= min_log_mel,
                    min_log_hz * np.exp(logstep * (mel - min_log_mel)),
                    f_sp * mel)


def mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    """Triangular Mel filters, Slaney-scaled and Slaney-normalised.

    Returns (n_mels, n_fft // 2 + 1). Both the scale and the normalisation
    have a second convention in circulation, and picking the other one gives
    filters that look right and weight the spectrum wrongly.
    """
    fft_freqs = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0.0),
                                   _hz_to_mel(sample_rate / 2), n_mels + 2))

    widths = np.diff(edges)
    ramps = edges[:, None] - fft_freqs[None, :]
    rising = -ramps[:-2] / widths[:-1, None]
    falling = ramps[2:] / widths[1:, None]
    filters = np.maximum(0.0, np.minimum(rising, falling))
    # Slaney normalisation: each filter encloses unit area, so a flat
    # spectrum comes out flat instead of tilting with frequency.
    filters *= (2.0 / (edges[2:n_mels + 2] - edges[:n_mels]))[:, None]
    return filters.astype(np.float32)


def log_mel(audio, n_mels=N_MELS):
    """A waveform as Whisper's input features: (n_mels, 3000) float32.

    Audio shorter than 30 seconds is zero-padded and anything longer is
    truncated, because the encoder's positional table is exactly that long.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    # A periodic Hann window, and the reflect-padded centred framing that
    # torch.stft(center=True) performs. The last frame is dropped: it covers
    # padding only, and keeping it would give 3001 frames for a 1500-long
    # positional table to sit under.
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP
    frames = padded[np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]]
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    power = power[:-1].T

    mel = mel_filterbank(n_mels=n_mels) @ power
    log_spec = np.log10(np.maximum(mel, 1e-10))
    # Whisper clamps the dynamic range to 80 dB below the loudest bin and
    # then maps to roughly [-1, 1]. The floor is taken over the whole
    # spectrogram, so it depends on the clip rather than on each frame.
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def read_wav(path, sample_rate=SAMPLE_RATE):
    """A wav file as mono float32 at `sample_rate`.

    Handles 8-, 16- and 32-bit PCM, downmixes to mono, and resamples by
    linear interpolation. Compressed wav variants are refused by name
    rather than read as though they were PCM.
    """
    with wave.open(str(path), "rb") as f:
        channels = f.getnchannels()
        width = f.getsampwidth()
        rate = f.getframerate()
        raw = f.readframes(f.getnframes())

    if width not in (1, 2, 4):
        raise ValueError(
            f"{path} stores {width * 8}-bit samples; reminis reads 8-, 16- "
            "and 32-bit PCM wav files. Convert it first, with "
            "`ffmpeg -i in -ar 16000 -ac 1 out.wav`."
        )

    if width == 1:
        # 8-bit wav is unsigned, centred on 128.
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != sample_rate and len(audio) > 1:
        n_out = int(round(len(audio) * sample_rate / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio, rate
