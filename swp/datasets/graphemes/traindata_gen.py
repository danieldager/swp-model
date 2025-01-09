import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .image_gen import get_all_fonts, get_dataset_max_font_size, text_to_grapheme


def random_cartesian_product(
    num_samples: int,
    word: str,
    fonts: list[str],
    global_rot: list[int],
    line_rot: list[int],
    letter_rot: list[int],
    sizes: list[int],
    spacing: list[int],
) -> list[dict]:
    r"""Sample `num_samples` different arg samples used for grapheme generation with the word `word`.
    Args are sampled without replacement from the cartesian product defined by :
    - all fonts in `fonts`
    - all global rotations (letter + line inclination) in `global_rot`
    - all line inclinations in `line_rot` (relative to drawn global rot)
    - all letter rotation (independent for every letter) from `letter rot` (relative to drawn global rot)
    - all font sizes in `sizes`
    - all letter spacings (independently for every bigram) from `spacing`
    - cases from all upper, all lower or Title (first letter in upper only)
    Returns a list of dict containing args to generate each sample image.
    """
    # by letter random case is not implemented
    amount = (
        len(fonts)
        * len(global_rot)
        * len(line_rot)
        * (len(letter_rot) ** len(word))
        * len(sizes)
        * (len(spacing) ** (len(word) - 1))
        * 3
    )

    def id_to_dict(id):
        items = {}
        items["word"] = word
        items["fontname"] = fonts[id % len(fonts)]
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
        items["size"] = sizes[id % len(sizes)]
        id //= len(sizes)
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

    draws = np.random.choice(amount, size=num_samples)
    return [id_to_dict(draw) for draw in draws]


def create_gen_arg_dict(path: Path, words: Sequence[str]):
    # TODO docstring
    train_path = path / "train"
    train_path.mkdir(exist_ok=True, parents=True)
    rotations = [-15, -10, -5, 0, 5, 10, 15]
    max_size, max_space = get_dataset_max_font_size(words)
    all_fonts = get_all_fonts()
    line_rot = [0]
    spaces = [
        0,
        int(max_space / 4),
        int(max_space / 2),
        int(3 * max_space / 4),
        max_space,
    ]
    sizes = [int(2 * max_size / 3), int(5 * max_size / 6), max_size]
    dataset_gen_dict = {
        "fonts": list(set(all_fonts)),
        "global_rotations": list(set(rotations)),
        "line_rot": list(set(line_rot)),
        "letter_rotations": list(set(rotations)),
        "spaces": list(set(spaces)),
        "sizes": list(set(sizes)),
    }
    gen_args_path = train_path / "gen_args.json"
    with gen_args_path.open("w") as f:
        json.dump(dataset_gen_dict, f)
    return dataset_gen_dict


def create_train_dataset(
    path: Path, words: Sequence[str], images_per_word: int
) -> None:
    r"""Create a grapheme dataset at `path / "train"` location.

    Creates `images_per_word` images per word in `words`, each saved in a directory named after the corresponding word.
    """
    dataset_gen_dict = create_gen_arg_dict(path, words)
    train_path = path / "train"
    for word in words:
        images_args = random_cartesian_product(
            num_samples=images_per_word,
            word=word,
            fonts=dataset_gen_dict["fonts"],
            global_rot=dataset_gen_dict["global_rotations"],
            line_rot=dataset_gen_dict["line_rot"],
            letter_rot=dataset_gen_dict["letter_rotations"],
            sizes=dataset_gen_dict["sizes"],
            spacing=dataset_gen_dict["spaces"],
        )
        word_dir = train_path / word
        word_dir.mkdir(parents=True, exist_ok=True)
        for arg in images_args:
            im = text_to_grapheme(**arg)
            im_name = f"{word}_{arg["fontname"]}_{arg["size"]}"
            im_name = f"{im_name}_l{arg["line_angle"]}"
            im_name = (
                f"{im_name}_charrot{"-".join(str(angle) for angle in arg["angles"])}"
            )
            im_name = f"{im_name}_sp{"-".join(str(space) for space in arg["spacing"])}"
            if not arg["case"][0]:
                case_name = "lowers"
            elif arg["case"][-1]:
                case_name = "uppers"
            else:
                case_name = "title"
            im_name = f"{im_name}_{case_name}"
            im_name = f"{im_name}.jpg"
            im.save(word_dir / im_name)


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


def get_gen_arg_dict(path) -> dict:
    r"""Return the arguments used to generate the dataset stored in `path / "train"` directory"""
    train_gen_args_path = path / "train" / "gen_args.json"
    with train_gen_args_path.open("r") as f:
        gen_arg_dict = json.load(f)
    return gen_arg_dict
