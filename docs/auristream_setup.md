# AuriStream Setup Notes

*Branch: `feat/auristream-setup` · Last updated: 2026-05-13*
*Status: research complete, smoke test NOT yet executed (models gated — see §Access)*

---

## 1. Source and availability

**Paper:** Tuckute, Kotar, Fedorenko, Yamins (2025).
"Representing Speech Through Autoregressive Prediction of Cochlear Tokens."
*Interspeech 2025.* arXiv: [2508.11598](https://arxiv.org/abs/2508.11598).

**Weights:** HuggingFace org `TuKoResearch/` — <https://huggingface.co/TuKoResearch>

**No dedicated GitHub repo found.** All model code lives in the HuggingFace repos
as `modeling_auristream.py` and `configuration_auristream.py`, loaded via
`trust_remote_code=True`.

### Access (GATED models)

The BigAudioDataset checkpoint series (e.g. `AuriStream100M_1Pred_BigAudioDataset_500k`)
and the WavCochCausalV8192 tokenizer are **gated**: you must log in to HuggingFace
and accept contact-info-sharing conditions before downloading.

```bash
huggingface-cli login    # paste your HF token
```

The `AuriStream1B_librilight_ckpt500k` checkpoint appears **public** (non-gated
— its model card fetched successfully without auth). Recommended entry point for
smoke tests.

Relevant model IDs:

| Model | ID | Gated? |
|---|---|---|
| WavCoch tokenizer | `TuKoResearch/WavCochV8192` | likely yes |
| WavCoch (causal variant) | `TuKoResearch/WavCochCausalV8192` | yes (returned 401) |
| AuriStream-100M (1 pred step) | `TuKoResearch/AuriStream100M_1Pred_BigAudioDataset_500k` | yes |
| AuriStream-100M (60 pred steps) | `TuKoResearch/AuriStream100M_60Pred_BigAudioDataset_500k` | yes |
| AuriStream-1B (LibriLight) | `TuKoResearch/AuriStream1B_librilight_ckpt500k` | **no** |

---

## 2. Install requirements

No new packages needed — already in `requirements.txt`:

```
torch>=2.5.1
torchaudio>=2.5.1
transformers>=4.40.0
```

`trust_remote_code=True` is required on every load call because the model uses
custom HuggingFace model classes not yet merged into the `transformers` library.

---

## 3. Model architecture

### WavCoch tokenizer

Biologically inspired cochlear tokenizer. Converts raw waveform → discrete cochlear
token IDs (integers 0–8191).

| Parameter | Value |
|---|---|
| Input sample rate | **16 kHz, mono** |
| FFT window size | 1001 samples |
| Hop length | **80 samples = 5 ms** |
| Frequency bins | 211 |
| Bottleneck | 13-bit LFQ (Learned Fast Quantization) |
| Vocabulary | 8 192 discrete tokens |
| Total parameters | 11.1 M |
| Token rate | **200 tokens/second** (empirical; 1000 tokens per 5 s) |

**Token count — empirical formula (verified synthetically and on real paradigm WAVs,
2026-05-13):**

```
L = floor(T / 80)    i.e. T // 80
```

Synthetic results:
- 5.00 s → 80 000 samples → **1000 tokens** (exact multiple)
- 1.00 s → 16 000 samples → **200 tokens** (exact multiple)
- 0.50 s →  8 000 samples → **100 tokens** (exact multiple)

Real paradigm WAV results (confirming floor for non-multiples):
- Press_FEMALE_C.wav → 13 056 samples → **163 tokens** (13056 / 80 = 163.2 → floor 163 ✓)
- Press_MALE_D.wav   → 13 440 samples → **168 tokens** (13440 / 80 = 168.0 → exact ✓)
- abosh_FEMALE_C.wav → 13 399 samples → **167 tokens** (13399 / 80 = 167.49 → floor 167 ✓)

**Floor behavior confirmed.** Trailing samples shorter than one 5 ms step (< 80 samples)
are not represented. This is relevant for phoneme boundary alignment: a phoneme that ends
mid-step will have its last partial step absent from the token sequence.

*Note: the paper describes 988 tokens for 5 s using a windowed cochleagram formula.
The HF model produces 1000 tokens for the same input. The exact internal alignment
is unverified from remote code; see §8 Q7. The practical external formula is `T // 80`.*

WavCoch input/output:

```
input  : (B, 1, T)   float32 waveform at 16 kHz
output : {'input_ids': (B, L)}   int64 token IDs,  L = n_tokens
```

### AuriStream GPT-style transformer

Causal autoregressive transformer predicting next cochlear token.

| | 100M | 1B |
|---|---|---|
| Parameters | 100.7 M | 970.1 M |
| Layers (`n_layer`) | 12 | 48 |
| Attention heads (`n_head`) | 12 | 16 |
| Hidden dim (`n_embd`) | 784* | 1 280 |
| Vocab | 8 192 | 8 192 |
| Context window | 4 096 tokens (~20 s) | 4 096 tokens |
| Position encoding | RoPE | RoPE |
| Normalization | RMSNorm (pre-norm) | RMSNorm |
| Activation | SiLU | SiLU |

*The paper reports 784 for 100M; the `AuriStreamConfig` default is 768. The actual
value in `config.json` of any specific checkpoint overrides this and must be verified
after loading (see §8 Unresolved questions).

---

## 4. Loading code

**Transformers version requirement:** WavCoch remote code is not compatible with
`transformers 5.x`. Use `transformers==4.57.6` (tested) or any `4.x` release.
`requirements.txt` is now pinned to `transformers>=4.40.0,<5`.

```bash
pip install "transformers==4.57.6"
```

```python
import torch
import torchaudio
from transformers import AutoModel

# Device: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# WavCoch tokenizer: audio → cochlear token IDs
quantizer = AutoModel.from_pretrained(
    "TuKoResearch/WavCochV8192",
    trust_remote_code=True,
).to(device).eval()

# AuriStream LM: token IDs → hidden states
#   Use AuriStream100M_1Pred_BigAudioDataset_500k once gated access is granted;
#   or AuriStream1B_librilight_ckpt500k (public) as an interim entry point.
lm = AutoModel.from_pretrained(
    "TuKoResearch/AuriStream1B_librilight_ckpt500k",
    trust_remote_code=True,
).to(device).eval()
```

---

## 5. Inference and hidden-state extraction

**Primary extraction target: the full layer-wise temporal hidden-state sequence.**
For each stimulus, extract and keep all `n_layer + 1` hidden-state tensors at their
native temporal resolution `(1, L, D)`. No pooling is applied at extraction time.
Phoneme-level averaging (see §6) happens in a later stage, after phoneme boundaries
are available from forced alignment.

### Step 1 — Preprocess audio

```python
wav, sr = torchaudio.load("stimulus.wav")  # shape: (C, T)
if wav.size(0) > 1:                        # stereo → mono
    wav = wav.mean(dim=0, keepdim=True)
if sr != 16_000:
    wav = torchaudio.transforms.Resample(sr, 16_000)(wav)
# wav: (1, T) at 16 kHz
```

### Step 2 — Tokenize with WavCoch

```python
with torch.no_grad():
    token_ids = quantizer(wav.unsqueeze(0).to(device))["input_ids"]
    # shape: (1, L)   where L = T // 80  (empirical; T in samples at 16 kHz)
```

### Step 3 — Extract full temporal hidden-state sequence from AuriStream

```python
with torch.no_grad():
    out = lm(token_ids, output_hidden_states=True)

# out.hidden_states : tuple of n_layer+1 tensors, each (B, L, D)
#   index 0 : initial token embedding
#   index k : hidden state after transformer block k  (k = 1 … n_layer)
#   index -1 : final hidden state (after last block + RMSNorm)
#
# out.logits : (B, L, 8192)  — only if output_logits=True

all_hidden = out.hidden_states    # tuple of (1, L, D) — full temporal sequence
# Keep all layers; do NOT pool over time here.
```

No forward hooks needed — the standard HuggingFace `output_hidden_states=True`
parameter exposes all layers natively.

---

## 6. Token ↔ time mapping

Token `i` corresponds to a nominal 5 ms grid cell:

```
time_start_s  = i * 80 / 16000  = i * 0.005       = i / 200
time_end_s    = (i + 1) * 0.005
time_center_s = (i + 0.5) * 0.005
```

**Empirically confirmed:** 200 tokens/second, token i starts at `i * 5 ms`. No initial
offset observed in the empirical output. The ~31 ms offset derived from the paper's
cochleagram window (1001 samples / 2) does not match the HF model's effective grid;
see §8 Q7 for the open question on exact internal alignment.

### Primary phoneme embedding strategy (mean pooling)

Once phoneme boundaries are available from forced alignment (e.g. MFA), the
phoneme embedding for phoneme `p` at layer `k` is computed as:

```
tokens_in_p = {i : i * 0.005 >= phoneme_start_s
                   AND i * 0.005 < phoneme_end_s}

phoneme_embedding[p, k] = mean over i in tokens_in_p of all_hidden[k][0, i, :]
                         # shape: (D,)
```

This gives one `(D,)`-vector per (stimulus, phoneme, layer). All tokens whose
window-start falls within the phoneme interval are included; the hidden-state
values are averaged across that time span.

This is the primary representation strategy. It is not computed during this
setup/feasibility branch — it requires phoneme boundaries from a forced aligner.

### Secondary analysis: last-token causal control (optional)

Because AuriStream is causal, the hidden state at position `i` accumulates context
from tokens 0…i and predicts token `i+1`. Using the last token of a phoneme
interval (`i = max(tokens_in_p)`) rather than mean pooling gives a single vector
that has seen all the phoneme's acoustic input. This can be used as a causal-control
comparison to mean pooling, but it is **not the primary representation**.

---

## 7. Smoke test status

**Script:** `scripts/audio/auristream_smoke_test.py`

### Run 1 — 2026-05-13: WavCoch import failure (transformers 5.5.0)

**Command run:**
```bash
python scripts/audio/auristream_smoke_test.py
```

**Failure:**
```
FATAL: model loading failed — cannot import name 'BatchEncoding' from
'transformers.tokenization_utils' (unknown location)
```

**Diagnosis:** HuggingFace auth and model download succeeded (config and remote
code files downloaded). The failure is a compatibility break: `modeling_wavcoch.py`
(remote code) does `from transformers.tokenization_utils import BatchEncoding`, but
in `transformers 5.5.0` (the installed version) `BatchEncoding` was moved to
`transformers.tokenization_utils_base`. The old import location no longer exposes it.

**Fix applied:** A local compatibility shim in `auristream_smoke_test.py` patches the
old location before `from_pretrained` executes the remote code:

```python
def _patch_transformers_tokenization() -> None:
    import transformers.tokenization_utils as _tu
    if not hasattr(_tu, "BatchEncoding"):
        from transformers.tokenization_utils_base import BatchEncoding
        _tu.BatchEncoding = BatchEncoding
```

Called at the top of `load_models()`, before `AutoModel.from_pretrained`.

**Scope:** This shim is scoped to this smoke test script only. It does not touch
the installed `transformers` package or any other pipeline code. It is a workaround
pending confirmation that the WavCoch remote code is updated for transformers ≥ 5.x.

**Status after fix:** Re-run attempted — see Run 2 below.

### Run 2 — 2026-05-13: WavCoch weights downloaded, second Transformers 5.x failure

**Command run:**
```bash
python scripts/audio/auristream_smoke_test.py
```

**Partial success:** WavCoch `config.json`, remote code files, and `model.safetensors`
(44.3 MB, 41/41 weights) all downloaded successfully. HuggingFace auth is OK.

**Failure:**
```
FATAL: model loading failed — 'WavCoch' object has no attribute 'all_tied_weights_keys'
```

**Diagnosis:** `all_tied_weights_keys` is a property introduced by `PreTrainedModel`
in a recent `transformers` version. The WavCoch remote code was written against an
older transformers API that did not require this attribute. Combined with the Run 1
`BatchEncoding` failure, this is a clear pattern: the WavCoch remote code is not
compatible with `transformers 5.5.0`.

**Recommended fix:** Downgrade to `transformers 4.57.6`, which is the most recent
4.x release and still satisfies the project requirement `transformers>=4.40.0`.
No code changes needed — the shim from Run 1 remains in place as a safety net.

**Status:** Resolved by downgrade — see Run 3.

### Run 3 — 2026-05-13: synthetic smoke test PASSED (transformers==4.57.6)

**Command:**
```bash
python -m pip install "transformers==4.57.6" && python scripts/audio/auristream_smoke_test.py
```

**Result: PASS.** WavCoch and AuriStream-1B both loaded. All three synthetic waveforms
processed successfully. Short clips handled natively (no forced 5 s padding).

**Confirmed facts:**

| Waveform | Samples | token_ids shape |
|---|---|---|
| 5.0 s | 80 000 | `(1, 1000)` |
| 1.0 s | 16 000 | `(1, 200)` |
| 0.5 s |  8 000 | `(1, 100)` |

- AuriStream-1B confirmed: `n_layer=48`, `n_head=16`, `n_embd=1280`, `vocab=8192`
- `output_hidden_states=True` returns **49 tensors**, each `(1, L, 1280)` — index 0 is
  the token embedding, indices 1–48 are transformer block outputs
- Final hidden state is finite; time dimension matches `token_ids` length
- Empirical token rate: **exactly 200 tokens/second** for all tested lengths

### Run 4 — 2026-05-13: real-audio smoke test PASSED

**Command:**
```bash
python scripts/audio/auristream_smoke_test.py --real-audio data/external/paradigm/raw/wav/
```

**Result: PASS.** Three real paradigm WAVs processed successfully. Short clips handled
natively. Floor rounding confirmed for non-multiple-of-80 inputs.

| File | Samples | token_ids shape | Notes |
|---|---|---|---|
| Press_FEMALE_C.wav | 13 056 | `(1, 163)` | 13056/80 = 163.2 → floor 163 ✓ |
| Press_MALE_D.wav   | 13 440 | `(1, 168)` | 13440/80 = 168.0 → exact ✓ |
| abosh_FEMALE_C.wav | 13 399 | `(1, 167)` | 13399/80 = 167.49 → floor 167 ✓ |

- `output_hidden_states=True` returns **49 tensors**, each `(1, L, 1280)` for all real inputs
- Final hidden states are finite for all real inputs
- Hidden-state time dimension exactly matches `token_ids` length for all real inputs
- Token rate consistent at **200 Hz** (5 ms/token)

**Both synthetic and real-audio smoke tests complete.** The setup branch has validated:
model access, loading, short-clip handling, floor-rounding behavior, hidden-state
availability, and the empirical 200 Hz token grid.

**Remaining caveat:** exact internal WavCoch cochleagram window alignment is not yet
confirmed from `modeling_wavcoch.py` remote code. For phoneme-boundary analysis the
practical formula `L = T // 80` is sufficient, but a final check of the remote code
is recommended before treating token indices as exact frame boundaries.

---

## 8. Unresolved questions

| # | Question | Priority | Status |
|---|---|---|---|
| 1 | Does WavCoch accept clips shorter than 5 s without padding? | HIGH | **RESOLVED** — yes, natively (5s→1000, 1s→200, 0.5s→100) |
| 2 | Exact `n_embd` for AuriStream-100M | HIGH | **RESOLVED for 1B**: n_embd=1280 confirmed. 100M untested (still needs gated access) |
| 3 | Correct WavCoch model name: `WavCochV8192` vs `WavCochCausalV8192`? | MEDIUM | Open — `WavCochV8192` used successfully |
| 4 | Is `WavCochCausalV8192` the right variant for our use? | MEDIUM | Open |
| 5 | Does `output_hidden_states` index 0 = token embedding? | LOW | **RESOLVED** — confirmed: index 0 is the embedding layer, indices 1–N_layer are block outputs |
| 6 | Alignment artifacts between WavCoch tokens and AuriStream hidden states? | LOW | Open — needs verification from real stimuli |
| 7 | Why does the empirical token count (L = T//80 = 1000 for 5 s) differ from the paper formula (988)? Exact internal WavCoch cochleagram alignment? | MEDIUM | Open — verify from `modeling_wavcoch.py` remote code |
| 8 | Token count for T not a multiple of 80 samples: floor or ceiling? | MEDIUM | **RESOLVED** — floor confirmed on real paradigm WAVs (13056→163, 13399→167) |

---

## 9. Technical risks

1. **Gating** — the most practical risk: extracting features from non-public model
   weights requires a logged-in HF account and acceptance of terms.

2. **5 s input assumption — RESOLVED.** WavCoch processes short clips natively
   without padding to 5 s. Token count scales linearly with duration (200 Hz grid).

3. **Non-multiple-of-80 inputs — RESOLVED.** WavCoch uses floor rounding: trailing
   samples shorter than one 5 ms step are dropped. The last partial step before a
   phoneme boundary may therefore be absent from the token sequence. For phoneme-
   boundary alignment, the practical rule is that token index `i` nominally covers
   `[i * 5ms, (i+1) * 5ms)`, and tokens whose window-start is within the phoneme
   interval are included in the mean pool.

4. **HF custom code stability** — `trust_remote_code=True` means the model code
   may change across commits. Pin to a specific revision once a working version
   is found.

5. **Memory** — AuriStream-1B with `output_hidden_states=True` returns 49 × `(1, L, 1280)`
   tensors. For L=988, each tensor ≈ 5 MB → 49 × 5 = ~245 MB for one forward pass.
   AuriStream-100M is much lighter: 13 × `(1, L, D)` ≈ 13 × ~1 MB = ~13 MB.

---

## 10. Next steps — extraction branch (`feat/auristream-extract-hidden-states`)

**Smoke tests passed.** The setup branch is complete. The next branch should be
`feat/auristream-extract-hidden-states`, branched from `feat/auristream-setup`.

Minimal changes needed for a proper extraction branch:

1. **Create `swp/audio/models/auristream.py`** implementing the `AudioModel` Protocol:
   - `__init__`: load WavCoch + AuriStream, store device
   - `sample_rate = 16_000`
   - `available_layers()`: return block index strings `"embedding"`, `"block_1"` …
     `"block_N"` (where N = `n_layer`); `"final"` as alias for `"block_N"`
   - `extract_activations(waveform, layers)`: resample to 16 kHz → tokenize (WavCoch)
     → forward with `output_hidden_states=True` → select requested layers
     → return `{layer: tensor of shape (D, L)}` (transposed from `(1, L, D)` to
     match the existing `[D, T]` convention of EnCodec/DAC wrappers)
   - No padding needed — WavCoch handles variable-length input natively
   - `reconstruct()`: not applicable; raise `NotImplementedError`

2. **Register under `"auristream"`** in the model registry.

3. **Update `sanity_check.py`** to accept `--model auristream`.

4. **Document layer naming convention** — unlike EnCodec/DAC (3 stable layers),
   AuriStream exposes `n_layer + 1` hidden states. For 1B: 49 layers. Decide
   whether to extract all or only selected layers (e.g. every 6th block for 1B).

5. **Verify exact WavCoch window alignment** from `modeling_wavcoch.py` remote code
   before treating token indices as precise frame boundaries.