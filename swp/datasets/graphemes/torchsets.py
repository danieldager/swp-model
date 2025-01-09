from bisect import bisect_right
from pathlib import Path
from string import ascii_letters
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms
from g2p_en import G2p
from torch.utils.data import ConcatDataset, DataLoader, Dataset, default_collate
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.imagenet import ImageNet

from ...utils.datasets import (
    get_epoch_numpy,
    get_phoneme_to_id,
    get_training_fold,
    get_val_fold,
)
from ...utils.paths import get_graphemes_dir, get_imagenet_dir
from .testdata_gen import check_test_dataset
from .traindata_gen import check_train_dataset

# TODO fix generators


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
        phoneme_to_id: dict[str, int],
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        g2p = G2p()

        # TODO rework to get max length beforehand
        length_to_pad = 10  # max_len + 5

        def to_phoneme(word: str) -> torch.Tensor:
            phonemes: list[str] = g2p(word)  # TODO rework to avoid calling g2p
            phonemes.append("<EOS>")
            phonemes.extend(["<PAD>" for _ in range(length_to_pad - len(phonemes))])
            # store in dict ?
            return torch.Tensor([phoneme_to_id[phoneme] for phoneme in phonemes])

        # TODO use word_to_phoneme dict and build like word to phon tensors

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
        `generator` : generator used to control random sampling
        other args are passed to the `ImageFolder` parent class

    Attributes :
        `class_to_sample_id` : dict mapping a class name to the set of sample ids of this class
        `fold_id` : index of loaded fold
        `train` : bool indicating if it is training split
        `id_tensor` : Tensor of size `[num_fold_classes, num_samples_per_class]` containing overall dataset index. First dim is indexed along the fold dataframe.
        `generator` : generator used to control random sampling
    """

    def __init__(
        self,
        root: Path,
        fold_id: int,
        train: bool,
        phoneme_to_id: dict[str, int],
        generator=None,
        transform: Callable[..., Any] | None = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Callable[[str], bool] | None = None,
        allow_empty: bool = False,
    ):
        super().__init__(
            root, phoneme_to_id, transform, loader, is_valid_file, allow_empty
        )
        self.fold_id = fold_id
        self.train = train
        if self.train:
            data_df = get_training_fold(self.fold_id)
            self.epoch_ids = get_epoch_numpy(self.fold_id)
        else:
            data_df = get_val_fold(self.fold_id)
            self.epoch_ids = np.arange(len(data_df))
        self.id_tensor = torch.stack(
            [
                torch.tensor(self.class_to_sample_id[class_name], dtype=torch.int)
                for class_name in data_df["Word"]
            ]
        )
        self.generator = generator

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        print(index)
        new_index = int(
            self.id_tensor[
                self.epoch_ids[index],
                torch.randint(self.id_tensor.shape[1], (1,), generator=self.generator),
            ].item()
        )
        return super().__getitem__(new_index)

    def __len__(self) -> int:
        return len(self.epoch_ids)


def get_grapheme_trainloader(fold_id: int, train: bool, batch_size: int) -> DataLoader:
    r"""Return a dataloader containing the grapheme training data corresponding to the `fold_id` fold, batched in size `batch_size`.

    Return the corresponding training data if `train` is set to `True`.
    Return the validation data otherwise.
    """
    check_train_dataset(get_graphemes_dir())
    grapheme_set = RandomizedFoldRepetitionDataset(
        root=get_graphemes_dir() / "train",
        fold_id=fold_id,
        train=train,
        phoneme_to_id=get_phoneme_to_id(),
        transform=torchvision.transforms.ToTensor(),
    )
    grapheme_loader = DataLoader(grapheme_set, batch_size)
    return grapheme_loader


def get_grapheme_testloader(batch_size: int) -> DataLoader:
    r"""Return a dataloader containing the grapheme test data batched in size `batch_size`."""
    check_test_dataset(get_graphemes_dir())
    grapheme_set = RepetitionDataset(
        root=get_graphemes_dir() / "test",
        phoneme_to_id=get_phoneme_to_id(),
        transform=torchvision.transforms.ToTensor(),
    )
    grapheme_loader = DataLoader(grapheme_set, batch_size)
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
    # TODO docstring
    batch_data = []
    batch_targets = [[] for i in range(num_tasks)]
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


def get_mixed_trainloader(fold_id: int, train: bool, batch_size: int) -> DataLoader:
    r"""Return a dataloader containing the grapheme training data corresponding to the `fold_id` fold, batched in size `batch_size`.

    Return the corresponding training data if `train` is set to `True`.
    Return the validation data otherwise.
    """
    # TODO update docstring
    check_train_dataset(get_graphemes_dir())
    grapheme_set = RandomizedFoldRepetitionDataset(
        root=get_graphemes_dir() / "train",
        fold_id=fold_id,
        train=train,
        phoneme_to_id=get_phoneme_to_id(),
        transform=torchvision.transforms.ToTensor(),
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
    train_loader = DataLoader(
        concat_dataset,
        batch_size=batch_size,
        collate_fn=lambda data: task_collate_fn(data, 2),
    )
    return train_loader
