import torch


def save_weights(filepath, embedding, encoder, decoder, epoch, checkpoint=None):
    if checkpoint:
        epoch = f"{epoch}_{checkpoint}"
    embedding_path = filepath / f"embedding{epoch}.pth"
    encoder_path = filepath / f"encoder{epoch}.pth"
    decoder_path = filepath / f"decoder{epoch}.pth"
    torch.save(embedding.state_dict(), embedding_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)


def load_weigths(filepath, embedding, encoder, decoder, epoch, device):
    embedding_path = filepath / f"embedding{epoch}.pth"
    encoder_path = filepath / f"encoder{epoch}.pth"
    decoder_path = filepath / f"decoder{epoch}.pth"
    embedding.load_state_dict(
        torch.load(embedding_path, map_location=device, weights_only=True)
    )
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    decoder.load_state_dict(
        torch.load(decoder_path, map_location=device, weights_only=True)
    )
