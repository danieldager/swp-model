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

from swp.test.repetition import beta_test
from swp.utils.paths import get_weights_dir
from swp.utils.datasets import get_test_data
from swp.utils.models import get_model, load_weights
from swp.utils.setup import seed_everything, set_device
from swp.datasets.phonemes import get_phoneme_testloader, get_sonority_dataset

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
        "--batch_size",
        type=int,
        default=512,
        help="Test dataloader batch size",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress in phonemes",
    )

    args = parser.parse_args()
    model_name = args.model_name
    training_name = args.training_name
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    include_stress = args.include_stress

    seed_everything()
    device = set_device()
    device = "cpu"  # TODO why do error out when using MPS ?

    if checkpoint is None:
        model_weights_dir = get_weights_dir() / model_name / training_name
        checkpoints = [f.stem.split(".")[-1] for f in model_weights_dir.glob("*.pth")]
    else:
        checkpoints = [checkpoint]

    for checkpoint in checkpoints:
        model = get_model(args.model_name)
        load_weights(
            model=model,
            checkpoint=checkpoint,
            model_name=model_name,
            training_name=training_name,
            device=device,
        )

        input_df = get_test_data()
        test_loader = get_phoneme_testloader(batch_size, include_stress)
        dfrs = beta_test(
            model=model,
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
        override_df = get_sonority_dataset(include_stress=include_stress)
        override_loader = get_phoneme_testloader(
            batch_size, include_stress, override_df
        )

        dfrs = beta_test(
            model=model,
            checkpoint=checkpoint,
            device=device,
            input_df=override_df,
            test_loader=override_loader,
            model_name=model_name,
            training_name=training_name,
            include_stress=include_stress,
            override_extra_str=override_extra_str,
        )
