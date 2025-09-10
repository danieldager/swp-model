import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from swp.datasets.graphemes import get_grapheme_trainloader, get_mixed_trainloader
from swp.models.autoencoder import Unimodel
from swp.models.decoders import DecoderLSTM, DecoderRNN
from swp.models.encoders import CorNetEncoder
from swp.models.losses import AuditoryXENT, FirstErrorXENT, TaskLosses
from swp.train.reading import train
from swp.utils.datasets import check_query, get_phoneme_to_id
from swp.utils.earlystop import SlurmHandler
from swp.utils.models import (
    get_model,
    get_model_name,
    get_train_args,
    get_train_name,
    load_last_training_checkpoint,
)
from swp.utils.setup import backend_setup, seed_everything, set_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name string, overrides other model parameters",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        default=None,
        help="Training name string, overrides other training parameters",
    )
    parser.add_argument(
        "--fold_id",
        type=str,
        default=None,
        help="Evaluation fold id",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size (fixed to 1 for repetition)",
    )
    parser.add_argument(
        "--recur_type",
        type=str,
        default="lstm",
        help="Recurrent network architecture : RNN or LSTM",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of recurrent subnetworks.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers in recurrent subnetworks for encoder and decoder",
    )
    parser.add_argument(
        "--cornet_model",
        type=str,
        default="Z",
        help="CORNet architecture to be used",
    )
    parser.add_argument(
        "--reading_hidden",
        type=int,
        default=128,
        help="Size of the reading head of CNN. Must match the total size of decoder hidden data",
    )
    parser.add_argument(
        "--from_model",
        type=str,
        default=None,
        help="Path to pre-trained decoder checkpoint",
    )  # TODO look at argcomplete for solution ?
    parser.add_argument(
        "--grapheme_only",
        action="store_true",
        help="Use only graphemes for training and not a mix of graphemes and ImageNet",
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
        default=0.0,
        help="Dropout rate for encoders and decoders",
    )
    parser.add_argument(
        "--tf_ratio",
        type=float,
        default=0.0,
        help="Teacher forcing ratio for decoder",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print logs during training",
    )
    parser.add_argument(
        "--load_all",
        action="store_true",
        help="Use tensor dataset instead of image dataset. Faster but might use significantly more RAM",
    )
    parser.add_argument(
        "--auto_requeue",
        action="store_true",
        help="Enable the program to requeue itself on SLURM clusters provided signal SIGUSR1 is sent early enough.",
    )
    parser.add_argument(
        "--train_part",
        type=str,
        help="Specifiy which part to train. Can be `all`, `hidden`, `encoder` or `decorder`. Hidden is included in encoder.",
    )
    args = parser.parse_args()
    seed_everything()
    backend_setup()
    device = set_device()

    # TODO add query to args ?
    query = "(Word.str.len() < 11) & (Word.str.len() > 1)"
    if query is not None:
        check_query(query=query)

    if args.train_name is None:
        batch_size = args.batch_size
        learn_rate = args.learn_rate
        fold_id = args.fold_id
        include_stress = args.include_stress
        loss = "classic"  # TODO add loss to arguments
        seed = args.seed
        train_part = args.train_part.lower()
        mixed = not args.grapheme_only
        train_name = get_train_name(
            batch_size,
            learn_rate,
            fold_id,
            include_stress,
            seed=seed,
            loss=loss,
            query=query,
            train_part=train_part,
            mixed=mixed,
        )
    else:
        train_name = args.train_name
        train_args = get_train_args(train_name)
        batch_size = train_args["batch_size"]
        learn_rate = train_args["learning_rate"]
        fold_id = train_args["fold_id"]
        include_stress = train_args["include_stress"]
        seed = train_args["seed"]
        loss = train_args["loss"]
        query = train_args["query"]
        train_part = train_args["train_part"]
        mixed = train_args["mixed"]

    phoneme_to_id = get_phoneme_to_id(include_stress)

    if args.model_name is None:
        recur_type = args.recur_type.upper()
        if recur_type not in ["RNN", "LSTM"]:
            raise ValueError("Invalid recurrent layer type")
        Decoder = DecoderRNN if recur_type == "RNN" else DecoderLSTM

        vocab_size = len(phoneme_to_id)
        encoder = CorNetEncoder(
            hidden_size=args.reading_hidden,
            cornet_model=args.cornet_model,
        )
        decoder = Decoder(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tf_ratio=args.tf_ratio,
        )
        model = Unimodel(encoder, decoder, start_token_id=phoneme_to_id["<SOS>"])
        model_name = get_model_name(model)
    else:
        model_name = args.model_name
        model = get_model(model_name)

    if args.from_model is not None:
        complete_dict: dict[str, Any] = torch.load(
            args.from_model, map_location=device, weights_only=True
        )
        decoder_dict = {
            k.split(".", 1)[1]: v
            for k, v in complete_dict.items()
            if k.startswith("decoder.")
        }
        model.decoder.load_state_dict(decoder_dict)

    match loss:
        case "classic":
            auditory_loss = AuditoryXENT()
        case "first":
            auditory_loss = FirstErrorXENT()
        case _:
            raise ValueError(
                "Argument of loss isn't recognized, use `classic` or `first`"
            )

    match train_part:
        case "all":
            optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        case "hidden":
            optimizer = optim.Adam(model.encoder.to_hidden.parameters(), lr=learn_rate)
        case "encoder":
            optimizer = optim.Adam(model.encoder.parameters(), lr=learn_rate)
        case "decoder":
            optimizer = optim.Adam(model.decoder.parameters(), lr=learn_rate)
        case _:
            raise ValueError(f"Part to train {train_part} not recognized")

    sig_handler = None
    trainset_generator = None
    trainloader_generator = None
    validset_generator = None
    validloader_generator = None
    last_epoch = 0
    if args.auto_requeue:
        sig_handler = SlurmHandler()
        (
            trainset_generator,
            trainloader_generator,
            validset_generator,
            validloader_generator,
            last_epoch,
        ) = load_last_training_checkpoint(
            model=model,
            optimizer=optimizer,
            model_name=model_name,
            train_name=train_name,
        )
        if args.verbose and last_epoch != 0:
            print(f"Successfully loaded training state at epoch {last_epoch}")

    if not mixed:
        train_loader = get_grapheme_trainloader(
            fold_id=fold_id,
            train=True,
            batch_size=batch_size,
            include_stress=include_stress,
            dataset_generator=trainset_generator,
            dataloader_generator=trainloader_generator,
            query=query,
            load_all=args.load_all,
        )
        valid_loader = get_grapheme_trainloader(
            fold_id=fold_id,
            train=False,
            batch_size=batch_size,
            include_stress=include_stress,
            dataset_generator=validset_generator,
            dataloader_generator=validloader_generator,
            query=query,
            load_all=args.load_all,
        )
        criterion = auditory_loss
    else:
        train_loader = get_mixed_trainloader(
            fold_id=fold_id,
            train=True,
            batch_size=batch_size,
            include_stress=include_stress,
            dataset_generator=trainset_generator,
            dataloader_generator=trainloader_generator,
        )
        valid_loader = get_mixed_trainloader(
            fold_id=fold_id,
            train=False,
            batch_size=batch_size,
            include_stress=include_stress,
            dataset_generator=validset_generator,
            dataloader_generator=validloader_generator,
        )
        criterion = TaskLosses([auditory_loss, nn.CrossEntropyLoss()])

    train(
        model=model,
        model_name=model_name,
        train_name=train_name,
        criterion=criterion,
        optimizer=optimizer,
        phoneme_to_id=phoneme_to_id,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=args.num_epochs,
        device=device,
        verbose=args.verbose,
        sig_handler=sig_handler,
        from_epoch=last_epoch,
    )
    if sig_handler is not None:
        sig_handler.land()
