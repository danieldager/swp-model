from typing import overload

import torch

## from ..models.autoencoder import Bimodel, Unimodel


def save_encdec_weights(filepath, encoder, decoder, epoch, checkpoint=None):
    # TODO delete, are kept for legacy compatibility
    if checkpoint is not None:
        epoch = f"{epoch}_{checkpoint}"
    encoder_path = filepath / f"encoder{epoch}.pth"
    decoder_path = filepath / f"decoder{epoch}.pth"
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)


def load_encdec_weigths(filepath, encoder, decoder, epoch, device):
    # TODO delete, are kept for legacy compatibility
    encoder_path = filepath / f"encoder{epoch}.pth"
    decoder_path = filepath / f"decoder{epoch}.pth"
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    decoder.load_state_dict(
        torch.load(decoder_path, map_location=device, weights_only=True)
    )


# def save_weights(filepath, model: Unimodel | Bimodel, epoch, checkpoint=None):
def save_weights(filepath, model, epoch, checkpoint=None):
    if checkpoint is not None:
        epoch = f"{epoch}_{checkpoint}"
    model_path = filepath / f"model_{epoch}.pth"
    torch.save(model.state_dict(), model_path)


# def load_weigths(filepath, model: Unimodel | Bimodel, epoch, device):
def load_weigths(filepath, model, epoch, device):
    model_path = filepath / f"model_{epoch}.pth"
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.bind()


def find_and_delete_batchdim(shape: torch.Size) -> tuple[int | None, torch.Size]:
    r"""Look for the dimension containing `-1`, and returns it as well as the shape
    without this dimension.
    """
    batch_dim = None
    purged_dims = []
    for i, dim_size in enumerate(shape):
        if dim_size == -1:
            if batch_dim is not None:
                raise ValueError(
                    f"Multiple dimensions corresponding to batch in shape {shape}"
                )
            batch_dim = i
        else:
            purged_dims.append(dim_size)
    return batch_dim, torch.Size(purged_dims)


def can_reshape_magic(
    initial_shape: torch.Size, expected_shapes: torch.Size | tuple[torch.Size, ...]
) -> bool:
    r"""Checks that a tensor of shape `initial_shape` can be resized or resized and split
    into tensor(s) of shape(s) `expected_shapes`.

    Dimensions containing `-1` are assumed to be batch dimensions.
    """
    init_batch_dim, purged_init_shape = find_and_delete_batchdim(initial_shape)
    num_init_units = 1
    for dim_size in purged_init_shape:
        num_init_units *= dim_size
    if isinstance(expected_shapes, torch.Size):
        num_hidden_units = 1
        hidden_batch_dim, purged_hidden_shape = find_and_delete_batchdim(
            expected_shapes
        )
        if not isinstance(hidden_batch_dim, type(init_batch_dim)):
            raise ValueError(
                f"Batch dimension mismatch between initial shape {initial_shape} and expected shape {expected_shapes}"
            )
        for dim_size in purged_hidden_shape:
            num_hidden_units *= dim_size
    else:
        num_hidden_units = 0
        for hidden_tensor_shape in expected_shapes:
            num_curr_hidden_units = 1
            curr_hidden_batch_dim, curr_purged_hidden_shape = find_and_delete_batchdim(
                hidden_tensor_shape
            )
            if not isinstance(curr_hidden_batch_dim, type(init_batch_dim)):
                raise ValueError(
                    f"Batch dimension mismatch between initial shape {initial_shape} and expected sub-shape {hidden_tensor_shape}"
                )
            for dim_size in curr_purged_hidden_shape:
                num_curr_hidden_units *= dim_size
            num_hidden_units += num_curr_hidden_units

    return num_init_units == num_hidden_units


def reshape_one(
    to_reshape: torch.Tensor,
    purged_expected_shape: torch.Size,
    expected_batch_dim: int | None,
) -> torch.Tensor:
    r"""Reshape `to_reshape` tensor (at least 2D if batched, with batch first) in
    shape `purged_batch_dim` then insert the batch dimension at the `expected_batch_dim` dimension.

    `purged_batch_dim` should NOT contain batch dimension.
    """
    if expected_batch_dim is not None:
        if to_reshape.dim() < 2:
            raise ValueError(
                f"Cannot infer batch dimension and data dimensions with only {to_reshape.dim()} dims"
            )
        reshaped = to_reshape.reshape(
            (to_reshape.size(0), *purged_expected_shape)
        )  # reshape all except batch dim
        end_permute = (
            [*range(1, expected_batch_dim + 1)]  # all the dimensions before batch dim
            + [0]  # then batch dim
            + [*range(expected_batch_dim + 1, len(purged_expected_shape) + 1)]
        )
        depermuted = reshaped.permute(end_permute)  # put batch dim where it should be
        to_ret = depermuted
    else:
        to_ret = to_reshape.reshape(purged_expected_shape)
    return to_ret


@overload
def reshape_magic(
    to_reshape: torch.Tensor,
    initial_shape: torch.Size,
    expected_shapes: torch.Size,
) -> torch.Tensor: ...


@overload
def reshape_magic(
    to_reshape: torch.Tensor,
    initial_shape: torch.Size,
    expected_shapes: tuple[torch.Size, ...],
) -> tuple[torch.Tensor, ...]: ...


def reshape_magic(
    to_reshape: torch.Tensor,
    initial_shape: torch.Size,
    expected_shapes: torch.Size | tuple[torch.Size, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Convert `to_reshape` tensor into tensor(s) of shape `expected_shapes`
    while conserving potential batch dimension. Splitting is done if
    `expected_shapes` is a tuple to generate the proper number of tensors.

    `initial_shape` should describe the shape of `to_reshape`, with a `-1` at
    the potential batch dimension.
    """
    init_batch_dim, _ = find_and_delete_batchdim(initial_shape)
    data_dim = 0
    if init_batch_dim is not None:
        init_permute = (
            [init_batch_dim]  # batch dim first
            + [*range(init_batch_dim)]  # then all the dimensions before batch dim
            + [*range(init_batch_dim + 1, to_reshape.dim())]  # then the rest
        )
        permuted = to_reshape.permute(init_permute)  # apply permutation
        to_reshape = permuted.flatten(start_dim=1)
        data_dim = 1

    if isinstance(expected_shapes, torch.Size):
        expected_batch_dim, purged_expected_shape = find_and_delete_batchdim(
            expected_shapes
        )
        to_ret = reshape_one(to_reshape, purged_expected_shape, expected_batch_dim)
    else:
        to_ret = []
        cumsum = 0
        for curr_expected_shape in expected_shapes:
            expected_batch_dim, purged_expected_shape = find_and_delete_batchdim(
                curr_expected_shape
            )
            length = 1
            for dim_size in purged_expected_shape:
                length *= dim_size
            curr_to_reshape = to_reshape.narrow(data_dim, cumsum, length)
            curr_hidden = reshape_one(
                curr_to_reshape, purged_expected_shape, expected_batch_dim
            )
            to_ret.append(curr_hidden)
            cumsum += length
        to_ret = tuple(to_ret)
    return to_ret
