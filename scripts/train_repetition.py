import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

import torch.nn as nn
import torch.optim as optim

from swp.datasets.phonemes import get_phoneme_trainloader
from swp.models.autoencoder import Unimodel
from swp.models.decoders import DecoderLSTM, DecoderRNN
from swp.models.encoders import EncoderLSTM, EncoderRNN
from swp.train.repetition import train
from swp.utils.datasets import get_phoneme_to_id
from swp.utils.models import (
    get_model,
    get_model_name,
    get_training_args,
    get_training_name,
)
from swp.utils.setup import seed_everything, set_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold_id",
        type=int,
        required=True,
        help="Evaluation fold id",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (fixed to 1 for repetition)",
    )
    parser.add_argument(
        "--learn_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--recurrent_type",
        type=str,
        default="rnn",
        help="Recurrent network architecture : RNN or LSTM",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers in recurrent subnetworks for encoder and decoder",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=-4,
        help="Hidden size of recurrent subnetworks.",
    )
    parser.add_argument(
        "--tf_ratio",
        type=float,
        default=0.2,
        help="Teacher forcing ratio for decoder",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for encoders and decoders",
    )

    args = parser.parse_args()
    seed_everything()
    device = set_device()

    # TODO mutually exclusive args
    if True:
        training_name = get_training_name(
            args.batch_size, args.learn_rate, args.fold_id
        )
        batch_size = args.batch_size
        learn_rate = args.learn_rate
        fold_id = args.fold_id
    else:
        training_name = args.training_name
        training_args = get_training_args(training_name)
        batch_size = training_args["b"]
        learn_rate = training_args["l"]
        fold_id = training_args["f"]

    # TODO mutually exclusive args
    if True:
        rec_type = args.recurrent_type.upper()
        phoneme_to_id = get_phoneme_to_id()
        vocab_size = len(phoneme_to_id)
        if rec_type == "LSTM":
            encoder = EncoderLSTM(
                vocab_size=vocab_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            decoder = DecoderLSTM(
                vocab_size=vocab_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                tf_ratio=args.tf_ratio,
            )
        elif rec_type == "RNN":
            encoder = EncoderRNN(
                vocab_size=vocab_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            decoder = DecoderRNN(
                vocab_size=vocab_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                tf_ratio=args.tf_ratio,
            )
        else:
            raise ValueError(
                f"Recurrent subnetwork {rec_type} is not recognized. Try RNN or LSTM."
            )
        model = Unimodel(
            encoder=encoder, decoder=decoder, start_token_id=phoneme_to_id["<SOS>"]
        )
        model_name = get_model_name(model)
    else:
        model_name = args.model_name
        model = get_model(model_name)

    train_loader = get_phoneme_trainloader(
        fold_id=fold_id,
        train=True,
        batch_size=batch_size,
        pad_to_length=0,
    )
    valid_loader = get_phoneme_trainloader(
        fold_id=fold_id,
        train=False,
        batch_size=batch_size,
        pad_to_length=0,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        training_name=training_name,
        num_epochs=args.num_epochs,
    )
