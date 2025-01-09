from pathlib import Path
from typing import Sequence

from .image_gen import text_to_grapheme
from .traindata_gen import get_gen_arg_dict


def exhaustive_cartesian_product(
    word: str,
    fonts: list[str],
    global_rot: list[int],
    line_rot: list[int],
    all_letter_rot: list[int],
    sizes: list[int],
    all_spacing: list[int],
) -> list[dict]:
    r"""Generate all possible arg samples used for grapheme generation with the word `word`.
    Possibilities are extracted from the cartesian product as follow :
    - all fonts in `fonts`
    - all global rotations (letter + line inclination) in `global_rot`
    - all line inclinations in `line_rot` (relative to drawn global rot)
    - all letter rotation (constant and equal for every letter) from `letter rot` (relative to drawn global rot)
    - all font sizes in `sizes`
    - all letter spacings (constant and equal for every bigram) from `spacing`
    - cases from all upper, all lower or Title (first letter in upper only)
    Returns a list of dict containing args to generate each image.
    """
    args = []
    for font in fonts:
        for global_rot_item in global_rot:
            for line_angle in line_rot:
                for letter_rot in all_letter_rot:
                    for size in sizes:
                        for space in all_spacing:
                            for case in [True, False, "Title"]:
                                if case == "Title":
                                    case_arg = [i == 0 for i in range(len(word))]
                                else:
                                    case_arg = [case for _ in word]
                                arg_dict = {
                                    "word": word,
                                    "fontname": font,
                                    "line_angle": line_angle + global_rot_item,
                                    "angles": [
                                        letter_rot + global_rot_item for _ in word
                                    ],
                                    "size": size,
                                    "spacing": [space for _ in word],
                                    "case": case_arg,
                                }
                                args.append(arg_dict)
    return args


def create_test_dataset(path: Path, words: Sequence[str]) -> None:
    r"""Create a grapheme dataset at `path / "test"` location.

    Number of images depends on the argument used to generate the training set.
    """
    train_gen_arg_dict = get_gen_arg_dict(path)
    test_path = path / "test"
    test_path.mkdir(exist_ok=True, parents=True)
    for word in words:
        images_args = exhaustive_cartesian_product(
            word=word,
            fonts=train_gen_arg_dict["fonts"],
            global_rot=train_gen_arg_dict["global_rotations"],
            line_rot=train_gen_arg_dict["line_rot"],
            all_letter_rot=train_gen_arg_dict["letter_rotations"],
            sizes=train_gen_arg_dict["sizes"],
            all_spacing=train_gen_arg_dict["spaces"],
        )
        word_dir = test_path / word
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


def check_test_dataset(path: Path) -> int:
    r"""Check that the number of images per word in the dataset located at `path / "test"`
    is constant and equal to the number of possibilities allowed by the arguments used to
    generate the training set, then return that number."""
    train_gen_arg_dict = get_gen_arg_dict(path)
    per_class_count = 3  # for UPPER, lower and Title casing
    per_class_count *= len(train_gen_arg_dict["fonts"])
    per_class_count *= len(train_gen_arg_dict["global_rotations"])
    per_class_count *= len(train_gen_arg_dict["line_rot"])
    per_class_count *= len(train_gen_arg_dict["letter_rotations"])
    per_class_count *= len(train_gen_arg_dict["sizes"])
    per_class_count *= len(train_gen_arg_dict["spaces"])
    path = path / "test"
    for dir in path.glob("*/"):
        num_files = len(
            list(dir.glob("**/*.jpg"))
        )  # TODO ensure it does not get trapped in an infinite cycle
        if per_class_count != num_files:
            raise RuntimeError(
                f"Number of images per class should be {per_class_count}, folder {dir} contains only {num_files} images"
            )
    return per_class_count
