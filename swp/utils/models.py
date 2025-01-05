import torch

from ..models.autoencoder import Bimodel, Unimodel


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


def save_weights(filepath, model: Unimodel | Bimodel, epoch, checkpoint=None):
    if checkpoint is not None:
        epoch = f"{epoch}_{checkpoint}"
    model_path = filepath / f"model_{epoch}.pth"
    torch.save(model.state_dict(), model_path)


def load_weigths(filepath, model: Unimodel | Bimodel, epoch, device):
    model_path = filepath / f"model_{epoch}.pth"
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.bind()
