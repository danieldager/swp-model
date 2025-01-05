import bisect
from math import ceil, sqrt
from pathlib import Path
from string import ascii_letters
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from g2p_en import G2p
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from ...utils.datasets import (
    get_epoch_numpy,
    get_phoneme_to_id,
    get_training_fold,
    get_val_fold,
)
from ...utils.paths import get_graphemes_dir

# TODO get_test_loader
