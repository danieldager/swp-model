import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

from swp.datasets.phonemes import get_phoneme_testloader, get_sonority_dataset
from swp.test.repetition import beta_test
from swp.utils.datasets import get_test_data
from swp.utils.models import get_model, load_weigths
from swp.utils.setup import seed_everything, set_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name string",
    )
    parser.add_argument(
        "--training_name",
        type=str,
        required=True,
        help="Training name string",
    )
    parser.add_argument(
        "--include_stress",
        # type=bool,
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Test dataloader batch size",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Epoch to load",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint to load",
    )

    args = parser.parse_args()

    model_name = args.model_name
    training_name = args.training_name
    include_stress = args.include_stress
    batch_size = args.batch_size
    epoch = args.epoch
    checkpoint = args.checkpoint

    seed_everything()
    device = set_device()
    device = "cpu"  # TODO why do error out when using MPS ?

    model = get_model(args.model_name)
    load_weigths(
        model_name=model_name,
        training_name=training_name,
        model=model,
        epoch=epoch,
        device=device,
        checkpoint=checkpoint,
    )

    input_df = get_test_data()
    test_loader = get_phoneme_testloader(batch_size, include_stress)
    dfrs = beta_test(
        model=model,
        epoch=epoch,
        checkpoint=checkpoint,
        device=device,
        input_df=input_df,
        test_loader=test_loader,
        model_name=model_name,
        training_name=training_name,
        include_stress=include_stress,
    )

    # Test also on the sonority dataset
    override_extra_str = "ssp"
    overide_df = get_sonority_dataset()
    overide_loader = get_phoneme_testloader(batch_size, include_stress, overide_df)

    dfrs = beta_test(
        model=model,
        epoch=epoch,
        checkpoint=checkpoint,
        device=device,
        input_df=overide_df,
        test_loader=overide_loader,
        model_name=model_name,
        training_name=training_name,
        include_stress=include_stress,
        override_extra_str=override_extra_str,
    )
