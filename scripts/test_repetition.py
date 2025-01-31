import os
import sys
import torch
import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The PyTorch API of nested tensors is in prototype stage",
)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.test.repetition import test
from swp.utils.models import get_model, load_weights
from swp.utils.setup import seed_everything, set_device
from swp.utils.paths import get_weights_dir, get_test_dir
from swp.utils.datasets import get_test_data, get_train_data
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
        "--train_name",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()
    model_name = args.model_name
    train_name = args.train_name
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    include_stress = args.include_stress

    seed_everything()
    device = set_device()
    device = torch.device("cpu")  # TODO why do error out when using MPS ?

    test_dir = get_test_dir()
    model_dir = test_dir / f"{model_name}~{train_name}"
    model_dir.mkdir(exist_ok=True, parents=True)
    weights_dir = get_weights_dir() / model_name / train_name

    if checkpoint is None:
        checkpoints = [f.stem.split(".")[-1] for f in weights_dir.glob("*.pth")]
    else:
        checkpoints = [checkpoint]

    for checkpoint in checkpoints:
        model = get_model(args.model_name)
        load_weights(
            model=model,
            model_name=model_name,
            train_name=train_name,
            checkpoint=checkpoint,
            device=device,
        )

        test_df = get_test_data()
        test_loader = get_phoneme_testloader(batch_size, include_stress)
        results, _ = test(
            model=model,
            device=device,
            test_df=test_df,
            test_loader=test_loader,
            include_stress=include_stress,
            verbose=args.verbose,
        )
        results.to_csv(model_dir / f"{checkpoint}.csv")

        # Test also on the sonority dataset
        ssp_df = get_sonority_dataset(include_stress=include_stress)
        ssp_loader = get_phoneme_testloader(batch_size, include_stress, ssp_df)
        results, _ = test(
            model=model,
            device=device,
            test_df=ssp_df,
            test_loader=ssp_loader,
            include_stress=include_stress,
            verbose=args.verbose,
        )
        results.to_csv(model_dir / f"{checkpoint}~ssp.csv")

        # Test also on the train dataset
        train_df = get_train_data()
        train_loader = get_phoneme_testloader(batch_size, include_stress, train_df)
        results, _ = test(
            model=model,
            device=device,
            test_df=train_df,
            test_loader=train_loader,
            include_stress=include_stress,
            verbose=args.verbose,
        )
        results.to_csv(model_dir / f"{checkpoint}~train.csv")

        if args.verbose:
            print("-" * 60)
