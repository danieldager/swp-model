import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

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


class RandomizedTensorReadingDataset(Dataset):
    r"""Dataset class to handle graphemes to phonemes dataset by loading the whole dataset in memory.
    Will track sample ids corresponding to the sample in the corresponding fold and store them in `id_tensor` attribute.

    Load the tensors located at `root`, and use `phoneme_to_id` for phoneme tokenization.

    Training fold is used if ̀`train` is set to ̀`True`, validation otherwise.

    Samples gotten from this dataset are randomized among the class they belong to.

    Args :
        `root` : root folder in which to look for class folders, containing sample images
        `fold_id` : fold number to load classes from
        `train` : return training split if set to `True`, validation split otherwise
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
        `include_stress` : if set to `True`, the phonemes will include stress
        `generator` : generator used to control random sampling. If `None`, then a generator is initialized deterministically.
        `query` : query to use when getting the data
        `sparse` : if set to `True`, will load the dataset as a sparse tensor

    Attributes :
        `img_tensor` : tensor containing all the image data
        `fold_id` : index of loaded fold
        `train` : bool indicating if it is training split
        `index_converter` : dict mapping index in dataframe to word ids
        `epoch_ids` : Array containing the class indices to go through over one epoch
        `generator` : generator used to control random sampling
        `query` : query used when getting the data
        `is_sparse` : bool indicating if `img_tensor` is a sparse tensor
        `phonemes` : dict mapping word ids to phoneme tokens tensors
        `max_len` : maximum length of a tokenized word
    """

    def __init__(
        self,
        root: Path,
        fold_id: int | None,
        train: bool,
        phoneme_to_id: dict[str, int],
        include_stress: bool = False,
        generator: torch.Generator | None = None,
        query: str | None = None,
        sparse: bool = True,
    ):
        self.fold_id = fold_id
        self.train = train
        self.query = query
        self.is_sparse = sparse
        if self.is_sparse:
            self.img_tensor: torch.Tensor = torch.load(root / "sparse_tensorset.pth")
        else:
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
            word_id = word_to_id[row["Word"]]
            self.index_converter[index] = word_id
            self.phonemes[word_id] = torch.Tensor(
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
        word_id = self.index_converter[self.epoch_ids[index]]
        if self.is_sparse:
            img = (
                1
                - self.img_tensor[
                    word_id,
                    int(
                        torch.randint(
                            self.img_tensor.size(1), (1,), generator=self.generator
                        )
                    ),
                ].to_dense()
            )
        else:
            img = self.img_tensor[
                word_id,
                int(
                    torch.randint(
                        self.img_tensor.size(1), (1,), generator=self.generator
                    )
                ),
            ]
        gt = self.phonemes[word_id]
        return img, gt

    def __len__(self) -> int:
        return len(self.epoch_ids)


class TensorReadingDataset(Dataset):
    r"""Dataset class to handle graphemes to phonemes dataset by loading the whole dataset in memory.
    Load the tensors located at `root`, and use `phoneme_to_id` for phoneme tokenization.

    Also implement a preprocessing for tokenizing the phonemes.

    Args:
        `root` : root folder in which to look for class folders, containing sample images
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
        `include_stress` : if set to `True`, the phonemes will include stress
        `query` : query to use when getting the data
        `sparse` : if set to `True`, will load the dataset as a sparse tensor

    Attributes:
        `img_tensor` : tensor containing all the image data
        `index_converter` : dict mapping index in dataframe to word ids
        `phonemes` : dict mapping word ids to phoneme tokens tensors
        `query` : query used when getting the data
        `is_sparse` : bool indicating if `img_tensor` is a sparse tensor
        `max_len` : maximum length of a tokenized word
    """

    def __init__(
        self,
        root: Path,
        phoneme_to_id: dict[str, int],
        include_stress: bool = False,
        query: str | None = None,
        sparse: bool = True,
    ):
        self.query = query
        self.is_sparse = sparse
        if self.is_sparse:
            self.img_tensor: torch.Tensor = torch.load(root / "sparse_tensorset.pth")
        else:
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
            word_id = word_to_id[row["Word"]]
            self.index_converter[index] = word_id
            self.phonemes[word_id] = torch.Tensor(
                [phoneme_to_id[phoneme] for phoneme in word_phonemes]
                + [phoneme_to_id["<EOS>"]]
            )
            max_len = max(max_len, len(word_phonemes) + 1)
        self.max_len = max_len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        real_id = index // self.img_tensor.shape[1]
        word_id = self.index_converter[real_id]
        img_id = index % self.img_tensor.shape[1]
        if self.is_sparse:
            img = 1 - self.img_tensor[word_id, img_id].to_dense()
        else:
            img = self.img_tensor[word_id, img_id]
        gt = self.phonemes[word_id]
        return img, gt

    def __len__(self) -> int:
        return len(self.index_converter) * self.img_tensor.shape[1]


class ReadingDataset(ImageFolder):
    r"""Dataset class to handle graphemes to phonemes dataset.
    Load the images located at `root`, and use `phoneme_to_id` for phoneme tokenization.

    Also implement a preprocessing for tokenizing the phonemes.

    Other arguments are passed to parent class.

    Args:
        `root` : root folder in which to look for class folders, containing sample images
        `word_to_phoneme` : dict mapping words to list of phonemes
        `phoneme_to_id` : dict mapping phonemes to int for tokenization
        other args are passed to the `ImageFolder` parent class

    Attributes:
        `class_to_sample_id` : dict mapping a class name to the set of sample ids of this class
        `max_len` : maximum length of a tokenized word
        `tokenized` : tuple containing tensors representing tokenized words
        Attributes from `ImageFolder` dataset class
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
        self.tokenized: tuple[torch.Tensor, ...]

        def to_phoneme(target: int) -> torch.Tensor:
            return self.tokenized[target]

        super().__init__(
            root=root,
            transform=transform,
            target_transform=to_phoneme,
            loader=loader,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.tokenized = tuple(
            torch.Tensor(
                [
                    phoneme_to_id[phoneme]
                    for phoneme in word_to_phoneme[self.classes[target]]
                ]
                + [phoneme_to_id["<EOS>"]]
            )
            for target in range(len(self.classes))
        )

        self.max_len = max(len(v) for v in word_to_phoneme.values()) + 1
        self.class_to_sample_id: dict[str, list[int]] = {}
        for sample_id, class_id in enumerate(self.targets):
            self.class_to_sample_id.setdefault(self.classes[class_id], []).append(
                sample_id
            )


class RandomizedFoldReadingDataset(ReadingDataset):
    r"""Subclass of `ReadingDataset` meant to handle folds.
    Stores a pairing between classes in the fold dataframe and image ids in `id_tensor` attribute.

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
        `fold_id` : index of loaded fold
        `train` : bool indicating if it is training split
        `id_tensor` : Tensor of size `[num_fold_classes, num_samples_per_class]` containing overall dataset index. First dim is indexed along the fold dataframe.
        `epoch_ids` : Array containing the class indices to go through over one epoch
        `generator` : generator used to control random sampling
        `query` : query used when getting the data
        Attributes from `ReadingDataset` class
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


class TaskConcatDataset(ConcatDataset):
    r"""Concatenate datasets with task names. Resulting dataset yields tuple `(data, target, task_name)`."""

    def __init__(self, tasks: dict[str, Dataset]) -> None:
        self.task_names = sorted(tasks.keys())
        super().__init__([tasks[key] for key in self.task_names])
        self.max_len = max(
            [getattr(dataset, "max_len", 0) for dataset in self.datasets]
        )

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
        return data, target, self.task_names[dataset_idx]
