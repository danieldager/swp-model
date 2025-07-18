import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torchvision.transforms
from torch.nested import nested_tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, default_collate
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.imagenet import ImageNet

from ...utils.datasets import (
    get_epoch_numpy,
    get_evaluation_dataset,
    get_phoneme_to_id,
    get_train_fold,
    get_valid_fold,
)
from ...utils.paths import get_graphemes_dir, get_imagenet_dir
from .testdata_gen import check_test_dataset
from .traindata_gen import check_train_dataset


class RandomizedTensorRepetitionDataset(Dataset):
    # TODO docstring
    def __init__(
        self,
        root: Path,
        fold_id: int | None,
        train: bool,
        phoneme_to_id: dict[str, int],
        include_stress: bool = False,
        generator: torch.Generator | None = None,
        query: str | None = None,
    ):
        self.fold_id = fold_id
        self.train = train
        self.query = query
        self.img_tensor: torch.Tensor = torch.load(root / "tensorset.pth")
        if self.train:
            data_df = get_train_fold(self.fold_id, query=self.query)
            self.epoch_ids = get_epoch_numpy(self.fold_id, query=self.query)
            print("train loop")
            print("epoch len : ", len(self.epoch_ids))
        else:
            data_df = get_valid_fold(self.fold_id, query=self.query)
            self.epoch_ids = np.arange(len(data_df))
        if include_stress:
            phoneme_label = "Phonemes"
        else:
            phoneme_label = "No_Stress"

        word_to_id_path = root / "word_to_id.json"
        with word_to_id_path.open("r") as f:
            word_to_id = json.load(f)

        self.index_converter = {}
        self.phonemes = {}
        max_len = 0
        for index, row in data_df.iterrows():
            word_phonemes = row[phoneme_label]
            tensor_id = word_to_id[row["Word"]]
            self.index_converter[index] = tensor_id
            self.phonemes[tensor_id] = torch.Tensor(
                [phoneme_to_id[phoneme] for phoneme in word_phonemes]
                + [phoneme_to_id["<EOS>"]]
            )
            max_len = max(max_len, len(word_phonemes) + 1)
        self.max_len = max_len
        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator().manual_seed(42)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        tensor_index = self.index_converter[self.epoch_ids[index]]
        img = self.img_tensor[
            tensor_index,
            int(torch.randint(self.img_tensor.size(1), (1,), generator=self.generator)),
        ]
        gt = self.phonemes[tensor_index]
        return img, gt

    def __len__(self) -> int:
        return len(self.epoch_ids)


class TensorRepetitionDataset(Dataset):
    # TODO docstring
    def __init__(
        self,
        root: Path,
        phoneme_to_id: dict[str, int],
        include_stress: bool = False,
        query: str | None = None,
    ):
        self.query = query
        self.img_tensor: torch.Tensor = torch.load(root / "tensorset.pth")
        data_df = get_evaluation_dataset(query=self.query)
        if include_stress:
            phoneme_label = "Phonemes"
        else:
            phoneme_label = "No_Stress"

        word_to_id_path = root / "word_to_id.json"
        with word_to_id_path.open("r") as f:
            word_to_id = json.load(f)

        self.index_converter = {}
        self.phonemes = {}
        max_len = 0
        for index, row in data_df.iterrows():
            word_phonemes = row[phoneme_label]
            tensor_id = word_to_id[row["Word"]]
            self.phonemes[tensor_id] = torch.Tensor(
                [phoneme_to_id[phoneme] for phoneme in word_phonemes]
                + [phoneme_to_id["<EOS>"]]
            )
            max_len = max(max_len, len(word_phonemes) + 1)
        self.max_len = max_len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        word_id = index // self.img_tensor.shape[1]
        img_id = index % self.img_tensor.shape[1]
        img = self.img_tensor[word_id, img_id]
        gt = self.phonemes[word_id]
        return img, gt

    def __len__(self) -> int:
        return self.img_tensor.shape[0] * self.img_tensor.shape[1]


class RepetitionDataset(ImageFolder):
    r"""Dataset class to handle graphemes to phonemes dataset.
    Load the images located at `root`, and use `phoneme_to_id` for phoneme tokenization.

    Also implement a preprocessing for tokenizing and padding the phonemes.

    Other arguments are passed to parent class.

    Args:
        `root` : root folder in which to look for class folders, containing sample images
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
        other args are passed to the `ImageFolder` parent class

    Attributes:
        `class_to_sample_id` : dict mapping a class name to the set of sample ids of this class
    """

    # is map-style dataset
    def __init__(
        self,
        root: Path,
        word_to_phoneme: dict[str, list[str]],
        phoneme_to_id: dict[str, int],
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):

        def to_phoneme(target: int) -> torch.Tensor:
            word = self.classes[target]
            phonemes = word_to_phoneme[word]
            return torch.Tensor(
                [phoneme_to_id[phoneme] for phoneme in phonemes]
                + [phoneme_to_id["<EOS>"]]
            )

        self.max_len = max(len(v) for v in word_to_phoneme.values()) + 1
        super().__init__(
            root,
            transform,
            to_phoneme,
            loader,
            is_valid_file,
            allow_empty,
        )
        self.class_to_sample_id: dict[str, list[int]] = {}
        for sample_id, class_id in enumerate(self.targets):
            self.class_to_sample_id.setdefault(self.classes[class_id], []).append(
                sample_id
            )


class RandomizedFoldRepetitionDataset(RepetitionDataset):
    r"""Subclass of `RepetitionDataset` meant to handle folds.
    Will track sample ids corresponding to the sample in the corresponding fold and store them in `id_tensor` attribute.

    Training fold is used if ̀`train` is set to ̀`True`, validation otherwise.

    Samples gotten from this dataset are randomized among the class they belong to.

    Args :
        `root` : root folder in which to look for class folders, containing sample images
        `fold_id` : fold number to load classes from
        `train` : return training split if set to `True`, validation split otherwise
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
        `generator` : generator used to control random sampling. If `None`, then a generator is initialized deterministically.
        `query` : query to use when getting the data
        other args are passed to the `ImageFolder` parent class

    Attributes :
        `class_to_sample_id` : dict mapping a class name to the set of sample ids of this class
        `fold_id` : index of loaded fold
        `train` : bool indicating if it is training split
        `id_tensor` : Tensor of size `[num_fold_classes, num_samples_per_class]` containing overall dataset index. First dim is indexed along the fold dataframe.
        `epoch_ids` : Array containing the class indices to go through over one epoch
        `generator` : generator used to control random sampling
        `query` : query used when getting the data
    """

    def __init__(
        self,
        root: Path,
        fold_id: int | None,
        train: bool,
        phoneme_to_id: dict[str, int],
        include_stress: bool = False,
        generator: torch.Generator | None = None,
        transform: Callable[..., Any] | None = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Callable[[str], bool] | None = None,
        allow_empty: bool = False,
        query: str | None = None,
    ):
        self.fold_id = fold_id
        self.train = train
        self.query = query
        if self.train:
            data_df = get_train_fold(self.fold_id, query=self.query)
            self.epoch_ids = get_epoch_numpy(self.fold_id, query=self.query)
        else:
            data_df = get_valid_fold(self.fold_id, query=self.query)
            self.epoch_ids = np.arange(len(data_df))
        if include_stress:
            phoneme_label = "Phonemes"
        else:
            phoneme_label = "No_Stress"
        word_to_phoneme = dict(zip(data_df["Word"], data_df[phoneme_label]))
        super().__init__(
            root,
            word_to_phoneme,
            phoneme_to_id,
            transform,
            loader,
            is_valid_file,
            allow_empty,
        )
        self.id_tensor = torch.stack(
            [
                torch.tensor(self.class_to_sample_id[class_name], dtype=torch.int)
                for class_name in data_df["Word"]
            ]
        )
        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator().manual_seed(42)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        new_index = int(
            self.id_tensor[
                self.epoch_ids[index],
                int(
                    torch.randint(
                        self.id_tensor.shape[1], (1,), generator=self.generator
                    )
                ),
            ].item()
        )
        return super().__getitem__(new_index)

    def __len__(self) -> int:
        return len(self.epoch_ids)


def get_grapheme_trainloader(
    fold_id: int | None,
    train: bool,
    batch_size: int,
    include_stress: bool = False,
    generator: torch.Generator | None = None,
    query: str | None = None,
    load_all: bool = True,
) -> DataLoader:
    r"""Return a dataloader containing the grapheme training data corresponding to the `fold_id` fold, batched in size `batch_size`.
    Shuffling is controlled by `generator`. If `generator` is None, it is deterministically instantiated.

    Return the corresponding training data if `train` is set to `True`.
    Return the validation data otherwise.

    Passing a `query` string gives a dataloader containing the corresponding queried dataset.
    """
    # TODO doc load_all
    if load_all:
        grapheme_set = RandomizedTensorRepetitionDataset(
            root=get_graphemes_dir() / "train",
            fold_id=fold_id,
            train=train,
            phoneme_to_id=get_phoneme_to_id(),
            include_stress=include_stress,
            query=query,
        )
    else:
        check_train_dataset(get_graphemes_dir())
        grapheme_set = RandomizedFoldRepetitionDataset(
            root=get_graphemes_dir() / "train",
            fold_id=fold_id,
            train=train,
            phoneme_to_id=get_phoneme_to_id(),
            include_stress=include_stress,
            transform=torchvision.transforms.ToTensor(),
            query=query,
        )
    if generator is None:
        generator = torch.Generator().manual_seed(42)
    pad_value = get_phoneme_to_id()["<PAD>"]
    my_collate = lambda batch: grapheme_collate_fn(batch, pad_value=pad_value)
    grapheme_loader = DataLoader(
        grapheme_set,
        batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=my_collate,
    )
    return grapheme_loader


def get_grapheme_testloader(
    batch_size: int,
    include_stress: bool = False,
    query: str | None = None,
    load_all: bool = True,
) -> DataLoader:
    r"""Return a dataloader containing the grapheme test data batched in size `batch_size`.
    Passing a `query` string gives a dataloader containing the corresponding queried test set.
    """
    # TODO doc load_all
    if load_all:
        grapheme_set = TensorRepetitionDataset(
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
        grapheme_set = RepetitionDataset(
            root=get_graphemes_dir() / "test",
            word_to_phoneme=word_to_phoneme,
            phoneme_to_id=get_phoneme_to_id(),
            transform=torchvision.transforms.ToTensor(),
        )
    pad_value = get_phoneme_to_id()["<PAD>"]
    my_collate = lambda batch: grapheme_collate_fn(batch, pad_value=pad_value)
    grapheme_loader = DataLoader(grapheme_set, batch_size, collate_fn=my_collate)
    return grapheme_loader


class IndicedConcatDataset(ConcatDataset):
    r"""Concatenate datasets. Resulting dataset yields tuple `(data, target, dataset_id)`."""

    def __init__(self, datasets: list[Dataset]) -> None:
        super().__init__(datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        data, target = self.datasets[dataset_idx][sample_idx]
        return data, target, dataset_idx


def task_collate_fn(
    batch: list[tuple[Any, Any, int]], num_tasks: int
) -> tuple[Any, tuple[list[Any], torch.Tensor]]:
    r"""This collate function is made to collate target tensors along the dataset they come from.
    It is meant to be used along the `IndicedConcacDataset` class.

    Returns a tuple containing :
      - a tensor of all the collated inputs
      - a list of tensors containing the collated target per corresponding dataset
      - a tensor containing the matching dataset id for the inputs
    """
    # TODO update to include nested tensors
    batch_data = []
    batch_targets = [[] for _ in range(num_tasks)]
    task_ids = []
    for sample in batch:
        data, target, id = sample
        batch_data.append(data)
        batch_targets[id].append(target)
        task_ids.append(id)
    batched_data = default_collate(batch_data)
    batched_targets = [default_collate(task_target) for task_target in batch_targets]
    batched_ids = default_collate(task_ids)
    return (batched_data, (batched_targets, batched_ids))


def grapheme_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_value: int):
    r"""A collate function that get tensors of different length from `batch`, then
    batch them together by extending them to the max length, filling with `pad_value`"""
    data, target = tuple(zip(*batch))
    collated_data = default_collate(list(data))
    nt_target = nested_tensor(list(target), dtype=torch.long)
    padded_target = nt_target.to_padded_tensor(padding=pad_value)
    return collated_data, padded_target


def get_mixed_trainloader(
    fold_id: int | None,
    train: bool,
    batch_size: int,
    include_stress: bool = False,
    generator: torch.Generator | None = None,
    query: str | None = None,
) -> DataLoader:
    r"""Return a dataloader containing both the grapheme training data corresponding to the `fold_id` fold
    and the ImageNet dataset, batched in size `batch_size`. Graphemes is the first dataset, ImageNet the second.
    Shuffling is controlled by `generator`. If `generator` is None, it is deterministically instantiated.

    Return the corresponding training data if `train` is set to `True`.
    Return the validation data otherwise.

    Passing a `query` string gives a dataloader containing the corresponding queried grapheme dataset.
    """
    check_train_dataset(get_graphemes_dir())
    grapheme_set = RandomizedFoldRepetitionDataset(
        root=get_graphemes_dir() / "train",
        fold_id=fold_id,
        train=train,
        phoneme_to_id=get_phoneme_to_id(),
        include_stress=include_stress,
        transform=torchvision.transforms.ToTensor(),
        query=query,
    )
    if train:
        imagenet_split = "train"
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
    concat_dataset = IndicedConcatDataset([grapheme_set, imagenet_set])
    if generator is None:
        generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        concat_dataset,
        batch_size=batch_size,
        collate_fn=lambda data: task_collate_fn(data, 2),
        shuffle=True,
        generator=generator,
    )
    return train_loader


def get_mixed_testloader(
    batch_size: int,
    include_stress: bool = False,
    query: str | None = None,
) -> DataLoader:
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
    grapheme_set = RepetitionDataset(
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
    concat_dataset = IndicedConcatDataset([grapheme_set, imagenet_set])
    test_loader = DataLoader(dataset=concat_dataset, batch_size=batch_size)
    return test_loader
