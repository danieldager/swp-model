import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms.functional as F

from .image_gen import ByFontArgs, get_gen_arg_dict, text_to_grapheme


def random_cartesian_product(
    num_samples: int,
    word: str,
    image_args: ByFontArgs,
    generator: np.random.Generator,
) -> list[dict]:
    r"""Sample `num_samples` different arg samples used for grapheme generation with the word `word`.
    Args are sampled without replacement from the cartesian product defined `image_args` :
    - all fonts in `font2sizes` dictionnary
    - all global rotations (letter + line inclination) in `global_rot`
    - all line inclinations in `line_rot` (relative to drawn global rot)
    - all letter rotation (independent for every letter) from `letter_rotations` (relative to drawn global rot)
    - all font sizes corresponding to the fonts
    - all letter spacings (independently for every bigram) from `spaces`
    - cases from all upper, all lower or Title (first letter in upper only)
    Returns a list of dict containing args to generate each sample image.
    Randomness is controlled by the `generator`.
    """
    fonts = list(image_args["font2sizes"].keys())
    global_rot = image_args["global_rotations"]
    line_rot = image_args["line_rot"]
    letter_rot = image_args["letter_rotations"]
    num_sizes = image_args["num_sizes"]
    spacing = image_args["spaces"]
    # by letter random case is not implemented
    amount = (
        len(fonts)
        * len(global_rot)
        * len(line_rot)
        * (len(letter_rot) ** len(word))
        * num_sizes
        * (len(spacing) ** (len(word) - 1))
        * 3
    )

    def id_to_dict(id):
        items = {}
        items["word"] = word
        font = fonts[id % len(fonts)]
        items["fontname"] = font
        id //= len(fonts)
        global_rot_item = global_rot[id % len(global_rot)]
        id //= len(global_rot)
        items["line_angle"] = line_rot[id % len(line_rot)] + global_rot_item
        id //= len(line_rot)
        letter_rots = []
        for _ in word:
            letter_rots.append(letter_rot[id % len(letter_rot)] + global_rot_item)
            id //= len(letter_rot)
        items["angles"] = letter_rots
        items["size"] = image_args["font2sizes"][font][id % num_sizes]
        id //= num_sizes
        spaces = []
        for _ in word[:-1]:
            spaces.append(spacing[id % len(spacing)])
            id //= len(spacing)
        items["spacing"] = spaces
        if id % 3 == 0:
            items["case"] = [True for _ in word]
        elif id % 3 == 1:
            items["case"] = [False for _ in word]
        else:
            items["case"] = [i == 0 for i in range(len(word))]
        return items

    draws = generator.choice(amount, size=num_samples, replace=False)
    return [id_to_dict(draw) for draw in draws]


def create_train_dataset(
    path: Path,
    words: Sequence[str],
    images_per_word: int,
    seed: int | None = None,
) -> None:
    r"""Create a grapheme dataset at `path / "train"` location.

    Creates `images_per_word` images per word in `words`, each saved in a directory named after the corresponding word.
    Use `seed` to control randomness. If none is provided, randomness is handled in a deterministic way.
    """
    if seed is None:
        seed = 42
    dataset_gen_dict = get_gen_arg_dict(path)
    train_path = path / "train"
    generator = np.random.default_rng(seed)
    for word in words:
        images_args = random_cartesian_product(
            num_samples=images_per_word,
            word=word,
            image_args=dataset_gen_dict,
            generator=generator,
        )
        word_dir = train_path / word
        word_dir.mkdir(parents=True, exist_ok=True)
        for arg in images_args:
            im = text_to_grapheme(**arg)
            im_name = f'{word}_{arg["fontname"]}_{arg["size"]}'
            im_name = f'{im_name}_l{arg["line_angle"]}'
            im_name = (
                f'{im_name}_charrot{"-".join(str(angle) for angle in arg["angles"])}'
            )
            im_name = f'{im_name}_sp{"-".join(str(space) for space in arg["spacing"])}'
            if not arg["case"][0]:
                case_name = "lowers"
            elif arg["case"][-1]:
                case_name = "uppers"
            else:
                case_name = "title"
            im_name = f"{im_name}_{case_name}"
            im_name = f"{im_name}.jpg"
            im.save(word_dir / im_name)


def create_train_tensor_dataset(
    path: Path,
    words: Sequence[str],
    images_per_word: int,
    seed: int | None = None,
) -> None:
    # TODO docstring
    if seed is None:
        seed = 42
    dataset_gen_dict = get_gen_arg_dict(path)
    train_path = path / "train"
    generator = np.random.default_rng(seed)
    dataset = torch.zeros((len(words), images_per_word, 3, 224, 224))
    order_tracker = {}
    word_to_id = {}
    for i, word in enumerate(sorted(words)):
        images_args = random_cartesian_product(
            num_samples=images_per_word,
            word=word,
            image_args=dataset_gen_dict,
            generator=generator,
        )
        for j, arg in enumerate(images_args):
            dataset[i, j] = F.to_tensor(text_to_grapheme(**arg))
        order_tracker[word] = images_args
        word_to_id[word] = i
    order_tracker_path = train_path / "order_tracker.json"
    with order_tracker_path.open("w") as f:
        json.dump(order_tracker, f, indent=4)
    word_to_id_path = train_path / "word_to_id.json"
    with word_to_id_path.open("w") as f:
        json.dump(word_to_id, f, indent=4)
    torch.save(dataset, train_path / "tensorset.pth")


def check_train_dataset(path: Path) -> int:
    r"""Check that the number of images per word in the dataset located at `path / "train"`
    is constant and non-zero, then return that number."""
    path = path / "train"
    counts = set()
    for dir in path.glob("*/"):
        num_files = len(
            list(dir.glob("**/*.jpg"))
        )  # TODO ensure it does not get trapped in an infinite cycle
        counts.add(num_files)
    if len(counts) == 1:
        num_images = counts.pop()
        if num_images != 0:
            return num_images
        else:
            raise RuntimeError("No images were found")
    else:
        raise RuntimeError(
            f"Number of images per class is not constant, different counts : {counts}"
        )
