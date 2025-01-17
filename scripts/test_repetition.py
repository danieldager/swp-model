import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse

from swp.datasets.phonemes import get_phoneme_testloader
from swp.test.repetition import beta_test
from swp.utils.datasets import get_test_data
from swp.utils.models import get_model, load_weigths
from swp.utils.setup import seed_everything, set_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name string"
    )
    parser.add_argument(
        "--training_name", type=str, required=True, help="Training name string"
    )
    parser.add_argument(
        "--include_stress",
        type=bool,
        action="store_true",
        help="Include stress in phonemes",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Test dataloader batch size"
    )
    parser.add_argument("--epoch", type=int, required=True, help="Epoch to load")
    parser.add_argument(
        "--checkpoint", type=int, default=None, help="Potential checkpoint to load"
    )

    parser.add_argument
    args = parser.parse_args()

    model_name = args.model_name
    training_name = args.training_name
    include_stress = args.include_stress
    batch_size = args.batch_size
    epoch = args.epoch
    checkpoint = args.checkpoint

    seed_everything()
    device = set_device()

    model = get_model(args.model_name)
    load_weigths(
        model_name=model_name,
        training_name=training_name,
        model=model,
        epoch=epoch,
        device=device,
        checkpoint=checkpoint,
    )

    override_data_df = None
    extra_str = None

    test_loader = get_phoneme_testloader(
        batch_size, include_stress, override_data_df=override_data_df
    )
    dfrs = beta_test(
        model=model,
        test_loader=test_loader,
        input_df=override_data_df if override_data_df is not None else get_test_data(),
        device=device,
        model_name=model_name,
        training_name=training_name,
        include_stress=include_stress,
        override_extra_str=extra_str,
    )
