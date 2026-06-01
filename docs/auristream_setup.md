# AuriStream Setup Notes

*Last updated: 2026-05-17*
*Status: setup complete; downstream pipeline fully implemented through PCA/norm analysis — see §7 and §10*

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
| Position encoding | Learned absolute PE (`wpe`)† | Learned absolute PE (`wpe`)† |
| Normalization | RMSNorm (pre-norm) | RMSNorm |
| Activation | SiLU | SiLU |

*The paper reports 784 for 100M; the `AuriStreamConfig` default is 768. The actual
value in `config.json` of any specific checkpoint overrides this and must be verified
after loading (see §8 Unresolved questions).

†`use_rope=False` is the `AuriStreamConfig` default and is **confirmed for the 1B checkpoint**
(`lm.config.use_rope = False`). When False, the model creates `wpe = nn.Embedding(seq_len, n_embd)`
(learned absolute positional embeddings) and sets `self.rotary = None` in every attention block.
RoPE is fully disabled. The 100M checkpoint is untested (gated access required).

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
#   index 0  : wte(token_id) + wpe(position)  — NOT a pure token embedding; see §11
#   index k  : output of block k−1 = input to block k  (k = 1 … 47)
#   index 48 : output of block 47 (last block), BEFORE final RMSNorm (ln_f); see §11
#
# WARNING: hidden_states[-1] is NOT post-ln_f. Apply manually if needed:
#   lm.transformer.ln_f(out.hidden_states[48])
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

### Primary phoneme embedding strategy (mean pooling, center-based selection)

The validated pipeline (`build_phoneme_embeddings.py`, default `--token-selection center`)
uses **center-based token selection**: token `i` is included if its center time falls
within the phoneme interval.

```
token_center_s(i) = (i + 0.5) / 200

tokens_in_p = {i : phoneme_start_s <= token_center_s(i) < phoneme_end_s}

phoneme_embedding[p, k] = mean over i in tokens_in_p of all_hidden[k][0, i, :]
                         # shape: (D,)
```

This gives one `(D,)`-vector per (stimulus, phoneme, layer).

Start-based selection (`--token-selection start`, using `i * 0.005` as the criterion)
is available as an alternative for reproducibility or causal-control comparisons,
but is not the default.

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
| 5 | Does `output_hidden_states` index 0 = token embedding? | LOW | **RESOLVED** — index 0 is `wte(token) + wpe(position)`, not a pure token embedding. Index 48 is raw output of last block **before** `ln_f`. See §11. |
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
   boundary alignment, the validated pipeline uses **center-based token selection**
   by default: token `i` is included if its center `(i + 0.5) / 200` falls within
   the phoneme interval `[start_s, end_s)`. Start-based selection (window-start
   within the interval) is available as an alternative/control via `--token-selection start`.

4. **HF custom code stability** — `trust_remote_code=True` means the model code
   may change across commits. Pin to a specific revision once a working version
   is found.

5. **Memory** — AuriStream-1B with `output_hidden_states=True` returns 49 × `(1, L, 1280)`
   tensors. For L=988, each tensor ≈ 5 MB → 49 × 5 = ~245 MB for one forward pass.
   AuriStream-100M is much lighter: 13 × `(1, L, D)` ≈ 13 × ~1 MB = ~13 MB.

---

## 10. Extraction branch — implemented and validated

**Branch:** `feat/auristream-extract-hidden-states` — **COMPLETE**

### Files implemented

| File | Role |
|---|---|
| `swp/audio/models/auristream.py` | AuriStream + WavCoch wrapper (`@register("auristream")`) |
| `swp/audio/pipeline/representation_extraction.py` | `run_representation_extraction()` — extraction-only pipeline (no signal metrics) |
| `scripts/audio/extract_representations.py` | CLI entry point |

### Validated full run

```bash
python scripts/audio/extract_representations.py \
    --model auristream \
    --dataset data/external/paradigm/processed/subset_male.csv \
    --layers embedding block_12 block_24 block_36 block_48 \
    --output reproduce/data/audio/
```

- **Run ID:** `auristream__9d3f269f`
- **Items:** 180 (full `subset_male.csv`)
- **Layers:** `embedding`, `block_12`, `block_24`, `block_36`, `block_48`
- **Tensor shape:** `[1280, L]` float32 CPU; L ∈ [119, 237], mean ≈ 168.5 tokens
- **Trailing samples:** max 78 (< 1 step = 5 ms; dropped by floor rounding, as expected)

### Extended run — `block_01`, `block_47`, `block_48_lnf` added

```bash
python scripts/audio/extract_representations.py \
    --model auristream \
    --dataset data/external/paradigm/processed/subset_male.csv \
    --layers embedding block_01 block_12 block_24 block_36 block_47 block_48 block_48_lnf \
    --output reproduce/data/audio/
```

- **Run ID:** `auristream__6ee9aeb6`
- **Items:** 180 (`subset_male.csv`)
- **Layers:** `embedding`, `block_01`, `block_12`, `block_24`, `block_36`, `block_47`, `block_48`, `block_48_lnf`
- **Phoneme embeddings:** 1055 phonemes × 8 layers, shape `[1055, 1280]` per layer

PCA results (phoneme-level, n = 1055):

| Layer | PC1 | PC2 |
|---|---|---|
| `embedding` | 14.4 % | 9.6 % |
| `block_01` | 16.6 % | 9.6 % |
| `block_12` | 11.5 % | 6.9 % |
| `block_24` | 12.6 % | 7.9 % |
| `block_36` | 11.6 % | 7.7 % |
| `block_47` | 11.9 % | 5.9 % |
| `block_48` | **82.9 %** | 2.9 % |
| `block_48_lnf` | 12.6 % | 5.6 % |

`block_48` PC1 = norm (r = 1.000, ρ = 1.000) — drops to 12.8 % after L2-normalization.
`block_48_lnf` PC1 not dominated by norm (r ≈ 0.355); L2-normalization leaves structure unchanged.
See §11 for full diagnostic.

### Remaining open question

Exact internal WavCoch cochleagram window alignment (§8 Q7) is still unverified from
`modeling_wavcoch.py` remote code. For the phoneme-embedding analysis, the practical
rule `token i → [i*5ms, (i+1)*5ms)` is sufficient until finer alignment is needed.

### Next step

Phoneme boundaries (MFA / forced alignment) → mean-pool hidden states per phoneme
interval → PCA / norm analyses.

**Downstream status (2026-05-17):** the full pipeline is implemented and validated
for `subset_male` across three branches:

| Step | Script |
|---|---|
| Hidden-state extraction | `scripts/audio/extract_representations.py` |
| MFA corpus + TextGrid → boundary CSV | `scripts/audio/prepare_mfa_corpus.py`, `textgrid_to_phoneme_boundaries.py` |
| Mean-pool → phoneme embeddings | `scripts/audio/build_phoneme_embeddings.py` |
| PCA + norm-by-position analysis | `scripts/audio/auristream_phoneme_pca_norm.py` |

See `docs/audio_project_state.md` §9 for validated run statistics and current
scientific status.

---

## 11. Embedding-layer and final-norm sanity checks

**Script:** `scripts/audio/auristream_embedding_sanity_checks.py`

Verified on `AuriStream1B_librilight_ckpt500k` using a synthetic repeated-token sequence
(token ID 42, length 64). No real audio required.

### What `hidden_states[0]` actually contains

For this checkpoint (`use_rope=False`, `dropout=0.0`), the identity holds exactly:

```
hidden_states[0][0, p, :] = wte.weight[token_id] + wpe.weight[p]
```

`wpe` is a **learned absolute positional embedding** (`nn.Embedding(4096, 1280)`). With
`dropout=0.0` there is no further transformation — the max abs error over 64 positions = **0.00e+00**.

**Consequence:** the layer called `embedding` in our pipeline is not a pure token representation.
It mixes token identity with absolute position, and norm variation across positions is
driven primarily by `||wpe(p)||`.

### Positional norm effect in the embedding layer

**Synthetic check** (same token repeated, 64 positions):

| Metric | Value |
|---|---|
| Pearson(‖wpe(p)‖, ‖hidden_states[0][p]‖) | **0.9959** |
| wte norm (single vector) | ≈ 30.2 |
| wpe norm range (positions 0–63) | ≈ 30–40 |
| h0 norm range | ≈ 42–50 |

**Confirmed on real phoneme embeddings** (`--phoneme-diagnostics`, run `auristream__6ee9aeb6`, 1055 phonemes):

| Correlation | Pearson r | Spearman ρ |
|---|---|---|
| `embedding_norm` vs `mean_wpe_norm` | **0.734** | **0.786** |
| `embedding_norm` vs `mean_wte_norm` | 0.264 | — |
| `embedding_norm` vs `mean_wte_wpe_cos` | 0.099 | — |

`mean_wte_norm` (≈ ‖wte[token_id_p]‖ averaged over the phoneme) is approximately flat across
phoneme positions. `mean_wpe_norm` follows the norm-by-position slope observed in `embedding_norm`.

**Conclusion:** norm variation in the `embedding` layer is primarily a signature of learned
positional embeddings (`wpe`), not of acoustic or phonemic token content.

### `hidden_states[48]` is before `ln_f` — and `block_48_lnf`

When calling `lm(seq, output_hidden_states=True)` without `output_logits=True`, the model
returns before applying `ln_f`. The full forward is:

```
hidden_states[48]  →  ln_f  →  coch_head  →  logits
```

Therefore:

- `out.hidden_states[48]` (`block_48`) = raw output of the last block, **before** `ln_f`
- `ln_f(out.hidden_states[48])` (`block_48_lnf`) = representation actually passed to `coch_head`

Effect of `ln_f` on `hidden_states[48]` (synthetic, 64 positions):

| Metric | Raw `block_48` | `block_48_lnf` |
|---|---|---|
| Norm spread (max − min) | **199.2** | **1.4** |
| Mean cosine similarity (raw vs post-ln_f) | — | **0.9974** |

`ln_f` strongly collapses norm variance while preserving direction.

### `block_48_lnf` in the extraction wrapper

`block_48_lnf` is exposed as a named layer in `swp/audio/models/auristream.py`
and returned by `available_layers()`. It is computed at token level inside the forward pass:

```python
# in extract_activations():
h = self._lm.transformer.ln_f(out.hidden_states[48])   # (1, L, D)
```

**Important:** phoneme embeddings for `block_48_lnf` are therefore:

```
mean_pool(ln_f(block_48 token states))   ≠   ln_f(mean_pool(block_48 token states))
```

RMSNorm is non-linear, so order matters. The stored phoneme embeddings are the former.

### PCA diagnostic: `block_48` vs `block_48_lnf`

Results on real phoneme embeddings (`auristream__6ee9aeb6`, 1055 phonemes):

| Layer | PC1 | PC2 | PC1 ~ norm (r) | After L2-norm: PC1 |
|---|---|---|---|---|
| `block_48` | **82.9 %** | 2.9 % | **1.000** | 12.8 % |
| `block_48_lnf` | 12.6 % | 5.6 % | 0.355 | ≈ 12.6 % |

In `block_48`, PC1 is entirely dominated by the L2 norm of the embedding (r = ρ = 1.000).
After L2-normalization, PC1 drops from 82.9 % to 12.8 %, revealing that the first principal
component carries no phonemic information — only magnitude.

In `block_48_lnf`, PC1 is not dominated by norm (r ≈ 0.355), and L2-normalization barely
changes the structure. This is the representation effectively sent to `coch_head`.

### Interpretation consequences

- **`embedding`** is `wte(token_id) + wpe(position)`, not a pure token representation.
  Norm variation with position is driven by `wpe`, confirmed on real phoneme data (r = 0.734).
- **Raw `block_48`** carries a large magnitude effect (PC1 = norm, r = 1.000) that `ln_f`
  neutralizes. Use `block_48` only for magnitude-specific diagnostics.
- **`block_48_lnf`** is the representation passed to `coch_head`; use it for phonemic geometry
  (PCA, RSA) and layer-depth comparisons.
- **Going forward:** interpret `embedding` as `wte + wpe`; prefer `block_48_lnf` over raw
  `block_48` for geometric analyses; keep raw `block_48` for norm diagnostics only.

### Phoneme-level diagnostic (`--phoneme-diagnostics`)

The script accepts `--phoneme-diagnostics` to run a second analysis on the real phoneme
embeddings, testing whether phoneme-level embedding norms are driven by `wpe` rather than
by acoustic token content.

```bash
# Synthetic checks only (checks 1–4):
python scripts/audio/auristream_embedding_sanity_checks.py

# Synthetic checks + phoneme-level diagnostic:
python scripts/audio/auristream_embedding_sanity_checks.py --phoneme-diagnostics
```

For each valid non-silence phoneme, it computes from saved activations (no re-extraction,
no re-tokenization with WavCoch):

| Column | Definition |
|---|---|
| `embedding_norm` | ‖mean_pool(h0)‖ — norm of the phoneme-level mean-pooled embedding |
| `mean_wpe_norm` | mean ‖wpe[p]‖ over token positions p inside the phoneme |
| `mean_wte_norm` | mean ‖h0[:,p] − wpe[p]‖ ≈ mean ‖wte[token_id_p]‖ |
| `mean_h0_tok_norm` | mean ‖h0[:,p]‖ over token positions p inside the phoneme |
| `mean_wte_wpe_cos` | mean cosine_similarity(wte[p], wpe[p]) per token, then averaged |

`mean_wte_norm` uses the algebraic identity `wte[token_id_p] = h0[:,p] − wpe[p]`, which
holds exactly (`dropout=0.0`, confirmed by check 2).

**Outputs** (default `reproduce/figures/audio/auristream_phonemes/auristream__9d3f269f/`):

| File | Content |
|---|---|
| `embedding_position_norm_diagnostics.csv` | Per-phoneme CSV with all columns above + metadata |
| `embedding_position_norm_summary.json` | Pearson + Spearman correlations, mean/std per norm |
| `embedding_position_norm_diagnostics.png` | Left: 3-curve norm-by-position (embedding, wpe, wte); Right: scatter |

**Confirmed results** (run `auristream__6ee9aeb6`, 1055 phonemes):

| Correlation | Pearson r | Spearman ρ |
|---|---|---|
| `embedding_norm` vs `mean_wpe_norm` | **0.734** | **0.786** |
| `embedding_norm` vs `mean_wte_norm` | 0.264 | — |
| `embedding_norm` vs `mean_wte_wpe_cos` | 0.099 | — |

`mean_wte_norm` is approximately flat across phoneme positions. `mean_wpe_norm` tracks the
norm-by-position slope observed in `embedding_norm`. The `wte–wpe` cosine alignment
(mean ≈ 0.099) contributes negligibly to the norm effect.

**Conclusion:** norm variation in the `embedding` layer over phoneme positions is driven by
`wpe` (r = 0.734), not by acoustic content (r = 0.264) or wte–wpe alignment (r = 0.099).

---

## 12. Cosine geometry and anisotropy diagnostics

*Run: `auristream__6ee9aeb6`*

### 12a. Scripts and output directories

| Script | Purpose | Output dir |
|--------|---------|-----------|
| `auristream_phoneme_cosine_distributions.py` | Phoneme-type cosine histograms (C-C/C-V/V-V), anisotropy, delta | `reproduce/figures/audio/auristream_phoneme_cosine/{run_id}/` |
| `auristream_wpe_position_diagnostics.py` | wpe norm-vs-position, propagation across layers | `reproduce/figures/audio/auristream_wpe_diagnostics/{run_id}/` |
| `auristream_token_vs_pooled_anisotropy.py` | Token-level vs phoneme-pooled anisotropy comparison | `reproduce/figures/audio/auristream_token_anisotropy/{run_id}/` |
| `auristream_verify_block48_lnf.py` | Verify `block_48_lnf = ln_f(block_48)` on saved activations | `reproduce/figures/audio/auristream_verify/{run_id}/` |
| `auristream_delta_position_diagnostics.py` | Position-specific diagnostics for `block_47 → block_48` delta | `reproduce/figures/audio/auristream_delta_position/{run_id}/` |

### 12b. Raw vs centered cosine

Two cosine modes are computed for all phoneme-type histograms:

| Mode | Definition | Use |
|------|-----------|-----|
| `raw` | L2-normalize each vector directly | Nafis-comparable; QC for loading/extraction |
| `centered` | Subtract per-dimension mean, then L2-normalize | Removes dominant common direction; main diagnostic when raw is saturated |

**Key finding:** raw cosine similarity is near-saturated at `block_48` and `block_48_lnf` — all C-C/C-V/V-V pairs have cosine ≈ 0.998. Centered cosine is necessary to reveal residual phoneme-type geometry.

### 12c. Anisotropy diagnostics

For each layer, the following metrics are computed (`--diagnose` flag):

| Metric | Definition | Primary indicator? |
|--------|-----------|-------------------|
| `isotropy_ratio` | `‖mean_vector‖ / mean(‖xᵢ‖)` — 0=isotropic, 1=collapsed | **Yes** |
| `cos_with_mean_mean` | Average cosine similarity of each vector with the mean direction | **Yes** |
| `cos_pairs_mean` | Average pairwise cosine (random sample) | **Yes** |
| `pca_pc1_evr` | PCA PC1 explained variance ratio | Supplementary only |

**Important:** `pca_pc1_evr` is NOT a reliable anisotropy indicator here. sklearn PCA centers internally; a dominant mean direction can saturate raw cosine without necessarily producing large centered PC1 variance. Use `isotropy_ratio` and `cos_with_mean_mean` as primary indicators.

Saved to: `anisotropy_diagnostics.csv`, `anisotropy_diagnostics_block47_block48_block48lnf.csv`

### 12d. block_48 anisotropy: findings

Raw-cosine anisotropy confirmed across all tested layers:

| Layer | isotropy_ratio | cos_pairs_mean (raw) | Interpretation |
|-------|---------------|---------------------|----------------|
| `embedding` | moderate | ≈ 0.28 | wpe-dominated norm; moderate direction spread |
| `block_24` | moderate | ≈ 0.51 | deepening contextual structure |
| `block_47` | high | ≈ 0.9+ | already anisotropic before last block |
| `block_48` | **≈ 0.999** | **≈ 1.00** | **extremely anisotropic** |
| `block_48_lnf` | **≈ 0.999** | **≈ 1.00** | anisotropy preserved after `ln_f` |

### 12e. block_48_lnf = ln_f(block_48) — verified

Script: `auristream_verify_block48_lnf.py`

**Result (3 items, max_abs_error = 0):** extraction is correct. `block_48_lnf` is exactly `ln_f(block_48)` at token level before phoneme pooling.

```bash
python scripts/audio/auristream_verify_block48_lnf.py \
    --run reproduce/data/audio/auristream__6ee9aeb6 \
    --max-items 3
```

### 12f. Token-level vs phoneme-pooled anisotropy

Script: `auristream_token_vs_pooled_anisotropy.py`

**Result:** anisotropy is consistent between token-level and phoneme-pooled levels. Isotropy ratios are comparable in both representations. Phoneme mean-pooling amplifies but does not create the phenomenon.

```bash
python scripts/audio/auristream_token_vs_pooled_anisotropy.py \
    --run reproduce/data/audio/auristream__6ee9aeb6 \
    --layers block_47 block_48 block_48_lnf
```

### 12g. Last-block delta: block_48 − block_47

The `--delta-diagnostic` flag in `auristream_phoneme_cosine_distributions.py` computes
`delta = block_48 − block_47` (phoneme-pooled level) with the following results:

| Metric | Value |
|--------|-------|
| `isotropy_ratio(delta)` | ≈ 0.9995 |
| `cos(delta_i, mean_delta)` | ≈ 0.9995 |
| `pairwise cos(delta_i, delta_j)` | ≈ 0.9990 |
| `cos(delta, block_47)` | ≈ 0.106 |
| `cos(delta, block_48)` | ≈ 0.992 |

**Working interpretation:** the last Transformer block adds a very large, near-common residual
update that dominates the direction of `block_48`. This is the primary cause of raw-cosine
saturation in `block_48` and `block_48_lnf`.

**Not yet determined:** whether this near-common delta is uniform across phoneme positions or
varies with absolute token position, causal context depth, or utterance structure. See §12h.

### 12h. Position-specific delta diagnostics

Script: `auristream_delta_position_diagnostics.py`

Tests whether the near-common `block_48 − block_47` delta depends on:
- `phoneme_position_from_start` (0, 1, 2, 3, 4, 5+ within item)
- phoneme type (consonant / vowel)
- absolute token midpoint position (early / mid / late quantile bins)
- token-level absolute position (Task B: `token_delta_by_absolute_position.csv`)

Cross-group alignment metric: `cos(group_mean_delta, global_mean_delta)` — if all groups ≈ 1,
the delta is globally uniform and not position-specific.

```bash
python scripts/audio/auristream_delta_position_diagnostics.py \
    --run reproduce/data/audio/auristream__6ee9aeb6 \
    --overwrite
```

**Interpretation guide:**
- All groups: high isotropy + cos(group_mean, global_mean) ≈ 1 → delta is global, not position-specific.
  The autoregressive / next-token-prediction interpretation remains possible but requires further evidence.
- Early positions (0–2) stronger than late → causal context depth or beginning-of-sequence effect.
- Token position explains better than phoneme position → temporal/autoregressive effect more likely.

### 12i. Recommended cosine analysis commands

```bash
# Full all-layer diagnostic run
python scripts/audio/auristream_phoneme_cosine_distributions.py \
    --run reproduce/data/audio/auristream__6ee9aeb6 \
    --layers embedding block_01 block_12 block_24 block_36 block_47 block_48 block_48_lnf \
    --analysis-families type \
    --cosine-modes raw centered \
    --max-samples-per-category 50000 \
    --diagnose \
    --delta-diagnostic \
    --overwrite

# Focused last-layer run (block_47 / block_48 / block_48_lnf)
python scripts/audio/auristream_phoneme_cosine_distributions.py \
    --run reproduce/data/audio/auristream__6ee9aeb6 \
    --layers block_47 block_48 block_48_lnf \
    --analysis-families type \
    --cosine-modes raw centered \
    --max-samples-per-category 50000 \
    --diagnose \
    --focused-last-layers \
    --delta-diagnostic \
    --overwrite

# Position-specific delta diagnostics
python scripts/audio/auristream_delta_position_diagnostics.py \
    --run reproduce/data/audio/auristream__6ee9aeb6 \
    --overwrite
```

### 12j. Block component diagnostics: attn_update vs MLP_update

**Script:** `scripts/audio/auristream_block_component_diagnostics.py`

Generic script that decomposes any Transformer block transition `input → output` into its
attention and MLP residual updates, using saved token-level activations (no re-extraction).

Architecture (Block.forward, normal path):
```python
x = x + attn_scale * attn(norm1(x))   # attn_scale = 1.0 in AuriStream-1B
x = x + mlp(norm2(x))
⟹ output = input + attn_update + mlp_update
```

Module paths (last block): `transformer.h[47].norm1/attn`, `transformer.h[47].norm2/mlp`, `transformer.ln_f`.

#### CLI

```bash
# Last block: block_47 → block_48
python scripts/audio/auristream_block_component_diagnostics.py \
    --run reproduce/data/audio/auristream__6ee9aeb6 \
    --block-index 47 --input-layer block_47 --output-layer block_48 \
    --max-items 30 --overwrite

# First block: embedding → block_01
python scripts/audio/auristream_block_component_diagnostics.py \
    --run reproduce/data/audio/auristream__6ee9aeb6 \
    --block-index 0 --input-layer embedding --output-layer block_01 \
    --max-items 30 --overwrite
```

Output dirs:
```
reproduce/figures/audio/auristream_block_components/auristream__6ee9aeb6/
  block_47_to_block_48/
  embedding_to_block_01/
```

Each transition folder contains:
`block_component_verification.json`, `block_component_verification.md`,
`component_anisotropy_token_level.csv`, `component_anisotropy_pooled_level.csv`,
`component_by_absolute_position.csv`, and 4 summary PNGs.

#### Reconstruction check (pass criterion: max_abs_error < 1e-3)

Both `output_recomp ≈ saved_output` and `input + attn_update + mlp_update ≈ saved_output` must pass.
For `block_47 → block_48`: reconstruction passes exactly (max_abs = 0).

#### Key scientific result — block_47 → block_48 (30 items confirmed)

| Metric | Value |
|--------|-------|
| `attn_update` mean norm / `delta_total` mean norm | **≈ 0.13** |
| `mlp_update`  mean norm / `delta_total` mean norm | **≈ 0.998** |
| cos(`attn_update`, `delta_total`) per-token mean  | **≈ 0.06** |
| cos(`mlp_update`,  `delta_total`) per-token mean  | **≈ 0.999** |
| `mlp_update` isotropy_ratio                       | **≈ 0.999** |
| `attn_update` isotropy_ratio                      | much lower  |

**Conclusion:** the near-common residual update in the last block is almost entirely carried
by the MLP, not by the attention. `attn_update` is small in norm and nearly orthogonal to
`delta_total`. This points to an **MLP-driven output-preparation** mechanism — possibly related
to the autoregressive next-token prediction objective — rather than a causal-context effect from
attention. The autoregressive / causal-context hypothesis would require the attention component
to dominate, which it does not.

#### Scientific caution

Do not claim this is *caused by* the autoregressive objective without further evidence.
The MLP in the last block may be specialised for projecting into the cochlear-token prediction
space, but this is a hypothesis. The `embedding → block_01` comparison (to be run) will test
whether the MLP-dominant common update is specific to the last block or a broader pattern.