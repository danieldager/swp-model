"""Low-level I/O for the frozen repeat model: encode a word, decode from a state.

These primitives are shared by the data layer (``can_repeat`` filters synthetic
edits) and the training layer (encode/decode each batch), so they live on their own
to keep ``core`` (data) and ``trainer`` (optimisation) decoupled.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def get_encoder_hidden(
    model: nn.Module, input_ids: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Final-layer encoder state ``(h, c)`` for a batch of token ids."""
    with torch.no_grad():
        h, c = model.encoder(input_ids.to(device))
    return h[-1], c[-1]


def decode_with_hidden(
    model: nn.Module,
    h: torch.Tensor,
    c: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    teacher_forcing: bool = False,
) -> torch.Tensor:
    """Autoregressively decode ``target.shape[1]`` steps from state ``(h, c)``.

    Returns logits of shape ``(batch, seq_len, vocab)``. With ``teacher_forcing`` the
    ground-truth token is fed at each step, otherwise the model's own prediction is.
    """
    h, c = h.unsqueeze(0), c.unsqueeze(0)
    batch_size, max_len = target.shape

    outputs: list[torch.Tensor] = []
    hidden = (h, c)
    inp = torch.full((batch_size, 1), model.start_token_id, device=device, dtype=torch.long)
    for t in range(max_len):
        embedded = model.decoder.dropout(model.decoder.embedding(inp))
        out, hidden = model.decoder.recurrent(embedded, hidden)
        logits = out @ model.decoder.embedding.weight.T
        outputs.append(logits)
        inp = target[:, t : t + 1] if teacher_forcing else logits.argmax(dim=-1)

    return torch.cat(outputs, dim=1)


def can_repeat(
    model: nn.Module,
    input_ids: list[int],
    device: torch.device,
    eos_id: int | None = None,
    pad_id: int = 0,
    max_len: int = 20,
) -> bool:
    """True if the frozen model decodes ``input_ids`` back to itself (free-running).

    With ``eos_id`` the word is encoded exactly as the training pipeline encodes it —
    ``seq + [EOS]`` padded to ``max_len`` — and the first ``len(seq)+1`` decoded steps
    must match. That is the same quantity ``count_correct`` scores, so a word this
    accepts is a word the intervention could in principle reach. Without ``eos_id`` the
    bare phoneme list is used, which tests a format the pipeline never actually feeds.
    """
    model.eval()
    seq = list(input_ids) if eos_id is None else list(input_ids) + [eos_id]
    padded = seq + [pad_id] * max(0, max_len - len(seq)) if eos_id is not None else seq
    ids = torch.tensor(padded[:max_len] if eos_id is not None else seq,
                       dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        h, c = get_encoder_hidden(model, ids, device)
        logits = decode_with_hidden(model, h, c, target=ids, device=device, teacher_forcing=False)
    pred = logits.argmax(dim=-1)[0, : len(seq)]
    return torch.equal(pred, ids[0, : len(seq)])
