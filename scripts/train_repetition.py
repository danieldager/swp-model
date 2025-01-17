import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

import torch.optim as optim

from swp.datasets.phonemes import get_phoneme_trainloader
from swp.models.autoencoder import Unimodel
from swp.models.decoders import DecoderLSTM, DecoderRNN
from swp.models.encoders import EncoderLSTM, EncoderRNN
from swp.models.losses import AuditoryXENT
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
        "--include_stress",
        type=bool,
        action="store_true",
        help="Include stress in phonemes",
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
        default=4,
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

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print logs during training",
    )

    args = parser.parse_args()
    seed_everything()
    device = set_device()

    # TODO group args
    # TODO mutually exclusive args
    if True:
        batch_size = args.batch_size
        learn_rate = args.learn_rate
        fold_id = args.fold_id
        include_stress = args.include_stress
        training_name = get_training_name(
            batch_size,
            learn_rate,
            fold_id,
            include_stress,
        )
    else:
        training_name = args.training_name
        training_args = get_training_args(training_name)
        batch_size = training_args["b"]
        learn_rate = training_args["l"]
        fold_id = training_args["f"]

    print(f"\nTraining name: {training_name}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Learning rate: {learn_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Teacher forcing ratio: {args.tf_ratio}")
    print(f"Dropout rate: {args.dropout}")

    # TODO mutually exclusive args
    # TODO printing arguments for debugging purposes
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
        include_stress=include_stress,
    )
    valid_loader = get_phoneme_trainloader(
        fold_id=fold_id,
        train=False,
        batch_size=batch_size,
        include_stress=include_stress,
    )
    criterion = AuditoryXENT()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # print all the parameters for debugging purposes
    print(f"Train_loader size: {len(train_loader)}")
    print(f"Valid_loader size: {len(valid_loader)}\n")

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
        verbose=args.verbose,
    )
