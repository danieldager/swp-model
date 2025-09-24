from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tensordict.nn import dispatch
from tensordict.tensordict import TensorDict

from ..utils.datasets import get_phoneme_to_id


class ErrorMeter(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()

    @abstractmethod
    def accumulate(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_errors(self) -> int | tuple[int, ...]:
        pass

    @abstractmethod
    def summary(self) -> str:
        pass


class SimpleErrorMeter(ErrorMeter):
    @abstractmethod
    def get_errors(self) -> int:
        pass


class FreeGenErrormeter(SimpleErrorMeter):
    r"""
    Returns the number of errors in the predicted phonemes while accounting
    for overgeneration
    """

    # TODO rework docstring
    @dispatch(source=["preds", "targets"])  # type:ignore
    def accumulate(self, tensordict):
        preds = tensordict["preds"]
        target = tensordict["targets"]
        self.errors += int(torch.any(preds != target, dim=1).sum().item())
        self.tot += target.shape[0]

    def reset(self):
        self.errors = 0
        self.tot = 0

    def get_errors(self):
        return self.errors

    def summary(self):
        return f"{self.errors}/{self.tot}"


class ClassicErrormeter(SimpleErrorMeter):
    r"""
    Returns the number of errors in the predicted phonemes, truncated by
    the length of the target phonemes
    """

    # TODO rework docstring
    @dispatch(source=["preds", "targets"])  #  type:ignore
    def accumulate(self, tensordict):
        preds = tensordict["preds"]
        target = tensordict["targets"]
        mask = target != get_phoneme_to_id()["<PAD>"]
        self.errors += int(torch.any((preds != target) * mask, dim=1).sum().item())
        self.tot += target.shape[0]

    def reset(self):
        self.errors = 0
        self.tot = 0

    def get_errors(self):
        return self.errors

    def summary(self):
        return f"{self.errors}/{self.tot}"


class ImageErrormeter(SimpleErrorMeter):
    # TODO docstring
    @dispatch(source=["preds", "targets"])  #  type:ignore
    def accumulate(self, tensordict):
        preds = tensordict["preds"]
        target = tensordict["targets"]
        self.errors += int((preds != target).sum().item())
        self.tot += target.shape[0]

    def reset(self) -> None:
        self.errors = 0
        self.tot = 0

    def get_errors(self) -> int:
        return self.errors

    def summary(self) -> str:
        return f"{self.errors}/{self.tot}"


class TaskErrormeter(ErrorMeter):
    # TODO docstring

    def __init__(self, sub_error_meters: dict[str, SimpleErrorMeter]) -> None:
        self.sub_error_meters = sub_error_meters
        super().__init__()

    def accumulate(self, tensordict: TensorDict):
        # check keys for each task
        for task_name, task_errormeter in self.sub_error_meters.items():
            task_errormeter.accumulate(tensordict[task_name])

    def reset(self):
        for sub_error_meter in self.sub_error_meters.values():
            sub_error_meter.reset()

    def get_errors(self):
        return tuple(
            sub_error_meter.get_errors()
            for sub_error_meter in self.sub_error_meters.values()
        )

    def summary(self):
        return "\n".join(
            [
                f"{task} : {sub_error_meter.summary}"
                for task, sub_error_meter in self.sub_error_meters.items()
            ]
        )
