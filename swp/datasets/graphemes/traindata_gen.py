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
    r"""Sample `num_samples` different (arg samples used for grapheme generation with the word `word`.
    Args are sampled without replacement from the cartesian product defined by :
    - all fonts in `fonts`
    - all global rotations (letter + line inclination) in `global_rot`
    - all line inclinations in `line_rot` (relative to drawn global rot)
    - all letter rotation (independent for every letter) from `letter rot` (relative to drawn global rot)
    - all font sizes in `sizes`
    - all letter spacing (independently for every bigram) from `spacing`
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


def create_train_dataset(
    path: Path, words: Sequence[str], images_per_word: int
) -> None:
    r"""Create a grapheme dataset at `Path` location.

    Creates `images_per_word` images per word in `words`, each saved in a directory named after the corresponding word.
    """
    rotations = [-15, -10, -5, 0, 5, 10, 15]
    max_size, max_space = get_dataset_max_font_size(words)
    all_fonts = get_all_fonts()
    spaces = [
        0,
        int(max_space / 4),
        int(max_space / 2),
        int(3 * max_space / 4),
        max_space,
    ]
    sizes = [int(2 * max_size / 3), int(5 * max_size / 6), max_size]
    for word in words:
        images_args = random_cartesian_product(
            num_samples=images_per_word,
            word=word,
            fonts=all_fonts,
            global_rot=rotations,
            line_rot=[0],
            letter_rot=rotations,
            sizes=sizes,
            spacing=spaces,
        )
        word_dir = path / word
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


def check_train_dataset(root: Path) -> int:
    r"""Check that the number of images per word in the dataset located at `root`
    is constant and non-zero, then return that number."""
    counts = set()
    for dir in root.glob("*/"):
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
