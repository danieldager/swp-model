import torch


def save_weights(filepath, encoder, decoder, epoch, checkpoint=None):
    if checkpoint:
        epoch = f"{epoch}_{checkpoint}"
    encoder_path = filepath / f"encoder{epoch}.pth"
    decoder_path = filepath / f"decoder{epoch}.pth"
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)


def load_weigths(filepath, encoder, decoder, epoch, device):
    encoder_path = filepath / f"encoder{epoch}.pth"
    decoder_path = filepath / f"decoder{epoch}.pth"
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    decoder.load_state_dict(
        torch.load(decoder_path, map_location=device, weights_only=True)
    )
