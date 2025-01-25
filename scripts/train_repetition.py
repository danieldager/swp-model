import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

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
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size (fixed to 1 for repetition)",
    )
    parser.add_argument(
        "--recur_type",
        type=str,
        default="rnn",
        help="Recurrent network architecture : RNN or LSTM",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden size of recurrent subnetworks.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers in recurrent subnetworks for encoder and decoder",
    )
    parser.add_argument(
        "--learn_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for encoders and decoders",
    )
    parser.add_argument(
        "--tf_ratio",
        type=float,
        default=0.2,
        help="Teacher forcing ratio for decoder",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
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

    # TODO mutually exclusive args
    # TODO printing arguments for debugging purposes
    if True:
        recur_type = args.recur_type.upper()
        phoneme_to_id = get_phoneme_to_id(include_stress=include_stress)
        vocab_size = len(phoneme_to_id)
        if recur_type == "LSTM":
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
        elif recur_type == "RNN":
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
                f"Recurrent subnetwork {recur_type} is not recognized. Try RNN or LSTM."
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

    if args.verbose:
        print("\n")
        print("-" * 60)
        print(f"\nModel name: {model_name}")
        print(f"Training name: {training_name}")
        print(f"Number of epochs: {args.num_epochs}")

    train(
        model=model,
        model_name=model_name,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        training_name=training_name,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=args.num_epochs,
        verbose=args.verbose,
    )

    if args.verbose:
        print("\n")
        print("-" * 60)
        print("\n")
