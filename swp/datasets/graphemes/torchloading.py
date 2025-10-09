from typing import Any, Callable

import torch
import torchvision.transforms
from tensordict.tensordict import TensorDict
from torch.nested import nested_tensor
from torch.utils.data import default_collate
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision.datasets.imagenet import ImageNet

from ...utils.datasets import get_evaluation_dataset, get_phoneme_to_id
from ...utils.paths import get_graphemes_dir, get_imagenet_dir
from .testdata_gen import check_test_dataset
from .torchsets import (
    RandomizedFoldReadingDataset,
    RandomizedTensorReadingDataset,
    ReadingDataset,
    TaskConcatDataset,
    TensorReadingDataset,
)
from .traindata_gen import check_train_dataset


def grapheme_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_value: int
) -> TensorDict:
    r"""A collate function that get tensors of different length from `batch`, then
    batch them together by extending them to the max length, filling with `pad_value`"""
    batch_dict = {}
    data, target = tuple(zip(*batch))
    batch_dict["inputs"] = default_collate(list(data))
    nt_target = nested_tensor(list(target), dtype=torch.long)
    padded_target = nt_target.to_padded_tensor(padding=pad_value)
    batch_dict["reading"] = {"targets": padded_target}
    return TensorDict(batch_dict)


def get_grapheme_trainloader(
    fold_id: int | None,
    train: bool,
    batch_size: int,
    include_stress: bool = False,
    dataset_generator: torch.Generator | None = None,
    dataloader_generator: torch.Generator | None = None,
    query: str | None = None,
    load_all: bool = False,
    num_workers: int = 0,
) -> StatefulDataLoader:
    r"""Return a dataloader containing the grapheme training data corresponding to the `fold_id` fold, batched in size `batch_size`.
    Shuffling is controlled by `dataloader_generator`. If `dataloader_generator` is None, it is deterministically instantiated.
    `dataset_generator` is passed to the dataset to control random sampling outside of shuffle.

    Return the corresponding training data if `train` is set to `True`.
    Return the validation data otherwise.

    Passing a `query` string gives a dataloader containing the corresponding queried dataset.
    Setting `load_all` to True returns a dataset loading everything in memory, which is faster to use but might be memory heavy.
    """
    if load_all:
        grapheme_set = RandomizedTensorReadingDataset(
            root=get_graphemes_dir() / "train",
            fold_id=fold_id,
            train=train,
            phoneme_to_id=get_phoneme_to_id(),
            include_stress=include_stress,
            generator=dataset_generator,
            query=query,
        )
    else:
        check_train_dataset(get_graphemes_dir())
        grapheme_set = RandomizedFoldReadingDataset(
            root=get_graphemes_dir() / "train",
            fold_id=fold_id,
            train=train,
            phoneme_to_id=get_phoneme_to_id(),
            include_stress=include_stress,
            generator=dataset_generator,
            transform=torchvision.transforms.ToTensor(),
            query=query,
        )
    if dataloader_generator is None:
        dataloader_generator = torch.Generator().manual_seed(42)
    pad_value = get_phoneme_to_id()["<PAD>"]
    my_collate = lambda batch: grapheme_collate_fn(batch, pad_value=pad_value)
    grapheme_loader = StatefulDataLoader(
        grapheme_set,
        batch_size,
        shuffle=True,
        generator=dataloader_generator,
        collate_fn=my_collate,
        pin_memory=True,
        num_workers=num_workers,
    )
    return grapheme_loader


def get_grapheme_testloader(
    batch_size: int,
    include_stress: bool = False,
    query: str | None = None,
    load_all: bool = False,
    num_workers: int = 0,
) -> StatefulDataLoader:
    r"""Return a dataloader containing the grapheme test data batched in size `batch_size`.
    Passing a `query` string gives a dataloader containing the corresponding queried test set.
    Setting `load_all` to True returns a dataset loading everything in memory, which is faster to use but might be memory heavy.
    """
    if load_all:
        grapheme_set = TensorReadingDataset(
            root=get_graphemes_dir() / "test",
            phoneme_to_id=get_phoneme_to_id(),
            include_stress=include_stress,
            query=query,
        )
    else:
        check_test_dataset(get_graphemes_dir())
        if include_stress:
            phoneme_label = "Phonemes"
        else:
            phoneme_label = "No_Stress"
        test_df = get_evaluation_dataset(query=query)
        word_to_phoneme = dict(zip(test_df["Word"], test_df[phoneme_label]))
        grapheme_set = ReadingDataset(
            root=get_graphemes_dir() / "test",
            word_to_phoneme=word_to_phoneme,
            phoneme_to_id=get_phoneme_to_id(),
            transform=torchvision.transforms.ToTensor(),
        )
    pad_value = get_phoneme_to_id()["<PAD>"]
    my_collate = lambda batch: grapheme_collate_fn(batch, pad_value=pad_value)
    grapheme_loader = StatefulDataLoader(
        grapheme_set,
        batch_size,
        collate_fn=my_collate,
        pin_memory=True,
        num_workers=num_workers,
    )
    return grapheme_loader


def phoneme_target_collate(targets: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    nt_target = nested_tensor(targets, dtype=torch.long)
    padded_target = nt_target.to_padded_tensor(padding=pad_value)
    return padded_target


def auto_target_collate_assigner(dataset: TaskConcatDataset) -> dict[str, Callable]:
    target_collates = {}
    for i, sub_dataset in enumerate(dataset.datasets):
        if isinstance(
            sub_dataset,
            (
                RandomizedFoldReadingDataset,
                RandomizedTensorReadingDataset,
                ReadingDataset,
                TensorReadingDataset,
            ),
        ):
            target_collates[dataset.task_names[i]] = (
                lambda targets: phoneme_target_collate(
                    targets, get_phoneme_to_id()["<PAD>"]
                )
            )
    return target_collates


def task_collate_fn(
    batch: list[tuple[Any, Any, str]],
    target_collates: dict[str, Callable],
) -> TensorDict:
    r"""This collate function is made to collate target tensors along the dataset they come from.
    It is meant to be used along the `IndicedConcacDataset` class.

    Returns a tuple containing :
      - a tensor of all the collated inputs
      - a list of tensors containing the collated target per corresponding dataset with the corresponding collate function
      - a tensor containing the matching dataset id for the inputs
    """
    batch_dict: dict[str, Any] = {"inputs": []}
    for i, sample in enumerate(batch):
        data, target, task_name = sample
        batch_dict["inputs"].append(data)
        batch_dict.setdefault(task_name, {}).setdefault("targets", []).append(target)
        batch_dict[task_name].setdefault("ids", []).append(i)
    batch_dict["inputs"] = default_collate(batch_dict["inputs"])
    for key in batch_dict:
        if key == "inputs":
            pass
        else:
            batch_dict[key]["targets"] = (
                target_collates[key](batch_dict[key]["targets"])
                if key in target_collates
                else default_collate(batch_dict[key]["targets"])
            )
            batch_dict[key]["ids"] = default_collate(batch_dict[key]["ids"])
    return TensorDict(batch_dict)


def get_mixed_trainloader(
    fold_id: int | None,
    train: bool,
    batch_size: int,
    include_stress: bool = False,
    dataset_generator: torch.Generator | None = None,
    dataloader_generator: torch.Generator | None = None,
    query: str | None = None,
    num_workers: int = 0,
) -> StatefulDataLoader:
    r"""Return a dataloader containing both the grapheme training data corresponding to the `fold_id` fold
    and the ImageNet dataset, batched in size `batch_size`. Graphemes is the first dataset, ImageNet the second.
    Shuffling is controlled by `dataloader_generator`. If `dataloader_generator` is None, it is deterministically instantiated.
    `dataset_generator` is passed to the grapheme dataset to control random sampling outside of shuffle.

    Return the corresponding training data if `train` is set to `True`.
    Return the validation data otherwise.

    Passing a `query` string gives a dataloader containing the corresponding queried grapheme dataset.
    """
    check_train_dataset(get_graphemes_dir())
    grapheme_set = RandomizedFoldReadingDataset(
        root=get_graphemes_dir() / "train",
        fold_id=fold_id,
        train=train,
        phoneme_to_id=get_phoneme_to_id(),
        include_stress=include_stress,
        generator=dataset_generator,
        transform=torchvision.transforms.ToTensor(),
        query=query,
    )
    if train:
        imagenet_split = "train"
        # TODO can we control the randomness without machine state ?
        imagenet_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        imagenet_split = "val"
        imagenet_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    imagenet_root = get_imagenet_dir()
    imagenet_set = ImageNet(
        imagenet_root,
        split=imagenet_split,
        transform=imagenet_transform,
    )
    task_dataset = TaskConcatDataset({"reading": grapheme_set, "recog": imagenet_set})
    if dataloader_generator is None:
        dataloader_generator = torch.Generator().manual_seed(42)
    target_collates = auto_target_collate_assigner(task_dataset)
    train_loader = StatefulDataLoader(
        task_dataset,
        batch_size=batch_size,
        collate_fn=lambda data: task_collate_fn(data, target_collates),
        shuffle=True,
        generator=dataloader_generator,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader


def get_mixed_testloader(
    batch_size: int,
    include_stress: bool = False,
    query: str | None = None,
    num_workers: int = 0,
) -> StatefulDataLoader:
    r"""Return a dataloader containing both the grapheme test data and the ImageNet
    test data, batched in size `batch_size`. Graphemes is the first dataset, ImageNet the second.

    Passing a `query` string gives a dataloader containing the corresponding queried grapheme dataset.
    """
    check_test_dataset(get_graphemes_dir())
    if include_stress:
        phoneme_label = "Phonemes"
    else:
        phoneme_label = "No_Stress"
    test_df = get_evaluation_dataset(query=query)
    word_to_phoneme = dict(zip(test_df["Word"], test_df[phoneme_label]))
    grapheme_set = ReadingDataset(
        root=get_graphemes_dir() / "test",
        word_to_phoneme=word_to_phoneme,
        phoneme_to_id=get_phoneme_to_id(),
        transform=torchvision.transforms.ToTensor(),
    )
    imagenet_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    imagenet_root = get_imagenet_dir()
    imagenet_set = ImageNet(
        imagenet_root,
        split="val",
        transform=imagenet_transform,
    )
    task_dataset = TaskConcatDataset({"reading": grapheme_set, "recog": imagenet_set})
    target_collates = auto_target_collate_assigner(task_dataset)
    test_loader = StatefulDataLoader(
        dataset=task_dataset,
        batch_size=batch_size,
        collate_fn=lambda data: task_collate_fn(data, target_collates),
        pin_memory=True,
        num_workers=num_workers,
    )
    return test_loader
