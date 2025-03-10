from typing import Iterable

import torch
import torch.nn as nn

from ..utils.datasets import get_phoneme_to_id


def alignment_loss(output, target, criterion, penalty):
    r"""
    Args:
        outputs: (output_len, vocab_size) tensor of logits
        targets: (target_len) tensor of target indices
    """
    # TODO Daniel more precise docstring
    output_len = output.size(0)
    target_len = target.size(0)

    # Initialize score matrix
    M = torch.zeros(output_len + 1, target_len + 1, device=output.device)

    # Initialize first row and column (penalty for skips)
    for i in range(output_len + 1):
        M[i, 0] = i * penalty
    for j in range(target_len + 1):
        M[0, j] = j * penalty

    # Fill matrix
    for i in range(1, output_len + 1):
        for j in range(1, target_len + 1):

            # Calculate match score using cross entropy
            score = criterion(output[i - 1].unsqueeze(0), target[j - 1].unsqueeze(0))

            # Take minimum of three possible operations:
            M[i, j] = torch.min(
                torch.stack(
                    [
                        M[i - 1, j - 1] + score,  # match/mismatch
                        M[i - 1, j] + penalty,  # skip in output
                        M[i, j - 1] + penalty,  # skip in target
                    ]
                )
            )

    # print("M[x,y]", M[output_len, target_len])
    # print("M", M)

    return M[output_len, target_len]


# Decoder forward pass using alignment loss ^^^
def alignment_forward(self, x, hidden, stop_token, target_len):
    # TODO Daniel docstring

    outputs = []

    # Set a limit for pred length
    max_length = target_len + 10

    # Forward pass loop
    for _ in range(max_length):
        output, hidden = self.rnn(x, hidden)

        # Generate output logits
        logits = self.fc(output)
        outputs.append(logits)

        # Check for stop token
        if torch.argmax(output) == stop_token:
            print("STOP")
            break

        # Pass output (not logits) to rnn
        x = output

    # Return logits (pred_len, vocab_size)
    outputs = torch.stack(outputs, dim=0)
    return outputs


class TaskLosses(nn.Module):
    r"""Apply different losses depending on the task in a same batch.
    `forward` expect the predictions to be a list of tensors, where `preds[i]`
    is the tensor of predictions for ith task.
    Also, the `targets` is a tuple containing the per-task `task_targets` (as a
    list of tensors) and a tensor containing the `task_ids` per sample.
    Does the weighted sum of `ith_loss(preds[i][task_ids == i], task_targets[i])`.

    Args :
        `losses` : iterable containing all losses, in order
        `weights` : Tensor containing weights to apply when summing the losses
    """

    def __init__(
        self, losses: Iterable[nn.Module], weights: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.task_losses = nn.ModuleList(losses)
        self.weights = weights
        if self.weights is not None and len(self.task_losses) != len(self.weights):
            raise ValueError("Number of losses and weights do not match")

    def forward(
        self,
        preds: list[torch.Tensor],
        targets: tuple[list[torch.Tensor], torch.Tensor],
    ):
        task_targets, task_ids = targets
        loss = 0
        for i in range(len(task_targets)):
            ith_task_loss = self.task_losses[i](
                preds[i][task_ids == i], task_targets[i]
            )
            if self.weights is not None:
                loss += self.weights[i] * ith_task_loss
            else:
                loss += ith_task_loss
        return loss


class AuditoryXENT(nn.CrossEntropyLoss):
    r"""Cross Entropy Loss made to automatically process the outputs of model classes
    like `Unimodel` or `Bimodel`.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(
            weight, size_average, ignore_index, reduce, reduction, label_smoothing
        )

    def forward(self, preds: list[torch.Tensor], targets: torch.Tensor):
        audit_preds = preds[0]
        audit_preds = audit_preds.flatten(end_dim=-2)
        targets = targets.flatten()
        targets = targets.clone()
        targets[targets == get_phoneme_to_id()["<PAD>"]] = -100
        return super().forward(audit_preds, targets)


class FirstErrorXENT(nn.CrossEntropyLoss):
    r"""Cross Entropy Loss made to automatically process the outputs of model classes
    like `Unimodel` or `Bimodel`.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(
            weight, size_average, ignore_index, reduce, reduction, label_smoothing
        )

    def forward(self, preds: list[torch.Tensor], targets: torch.Tensor):
        audit_preds = preds[0]
        targets = targets.clone()
        mismatches = audit_preds.argmax(dim=-1) != targets
        first_error = mismatches.argmax(dim=-1, keepdim=True)

        mask = torch.arange(targets.shape[-1], device=targets.device)
        if len(targets.shape) > 1:
            mask = mask.unsqueeze(0)
        mask = mask > first_error
        targets[mask] = -100

        audit_preds = audit_preds.flatten(end_dim=-2)
        targets = targets.flatten()

        return super().forward(audit_preds, targets)
