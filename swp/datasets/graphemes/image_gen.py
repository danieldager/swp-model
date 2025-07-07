import json
from pathlib import Path
from string import ascii_letters
from typing import Sequence, TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...utils.paths import get_fonts_dir

SCRIPT_FONTS = ["brushscriptstd", "Pacifico-Regular"]
SANS_FONTS = ["Arial", "helvetica"]
SERIF_FONTS = ["Times_New_Roman", "Georgia"]
_font_angle_cache: dict[tuple[str, str, int], tuple[int, float]] = {}
_optimized_angle_width_cache: dict[frozenset[int], dict[tuple[str, str, int], int]] = {}


def get_all_fonts() -> list[str]:
    r"""Returns all fonts used for image generation"""
    return SERIF_FONTS + SANS_FONTS + SCRIPT_FONTS


def text_to_grapheme(
    word: str,
    fontname: str,
    W: int = 224,
    H: int = 224,
    size: int = 20,
    spacing: list[int] | int = [],
    angles: list[int] | int = [],
    case: list[bool] | str = [],
    line_angle: int = 0,
    xshift: int = 0,
    yshift: int = 0,
) -> Image.Image:
    r"""Generate an image of size `WxH` where `word` is written in black over a white background.

    Writing font is controlled by `fontname` and `size`, provided `{fontname}.ttf` is an available True Type Font file.

    The word is centered over `(xshift, yshift)` and is written over a line inclined with an angle `line_angle` (in degrees).

    Per char case is controlled through `case` argument. Missing values will default to lower.

    Letter rotations (in degrees) can be controlled with ̀ angles`. Missing values will default to 0.

    Spacing between letters (in pixel) can be controlled through `spacing`. Missing values will default to 0.
    """

    if isinstance(spacing, int):
        spacing = [spacing for _ in range(len(word) - 1)]
    elif len(spacing) != len(word):
        spacing = spacing[: (len(word) - 1)]
        spacing += [0 for _ in range(len(word) - len(spacing))]
    if isinstance(angles, int):
        angles = [angles for _ in word]
    elif len(angles) != len(word):
        angles = angles[: (len(word))]
        angles += [0 for _ in range(len(word) - len(angles))]
    if isinstance(case, str):
        if case == "upper":
            case = [True for _ in word]
        elif case == "lower":
            case = [False for _ in word]
        elif case == "title":
            case = [i == 0 for i in range(len(word))]
        else:
            raise ValueError(
                "Case string argument not recognized, try with upper, lower or title"
            )
    elif len(case) != len(word):
        case = case[: (len(word))]
        case += [False for _ in range(len(word) - len(case))]
    word = "".join(
        [word[i].upper() if case[i] else word[i].lower() for i in range(len(word))]
    )

    img = Image.new("RGB", (W, H), color="white")
    fnt = ImageFont.truetype(get_fonts_dir() / f"{fontname}.ttf", size)
    letter_width = []
    letter_height = []
    letter_patches = []
    letter_offset = []

    # precompute rotated letters dimensions
    for i, l in enumerate(word):
        left, top, right, bottom = fnt.getbbox(l)
        width = right - left
        height = bottom - top
        txt = Image.new("RGBA", (int(width), int(height)), color=(0, 0, 0, 0))
        d = ImageDraw.Draw(txt)
        d.text((0, 0 - top), l, font=fnt, fill="black")
        txt = txt.rotate(angles[i], expand=True)
        letter_width.append(txt.width)
        letter_patches.append(txt)
        letter_offset.append(bottom - txt.height)
        letter_height.append(txt.height)

    # Starting word anchor
    h = max(letter_height) - min(letter_offset)
    w = sum(letter_width) + sum(spacing)
    anchor_vector = np.array([[-w / 2], [-h / 2]])
    origin = np.array([[W / 2], [H / 2]])
    shift = np.array([[xshift], [yshift]])
    rad_angle = line_angle / 180 * np.pi
    rot_mat = np.array(
        [
            [np.cos(rad_angle), np.sin(rad_angle)],
            [-np.sin(rad_angle), np.cos(rad_angle)],
        ]
    )

    letter_origin = origin + rot_mat @ anchor_vector + shift
    original_x = letter_origin[0, 0]
    original_y = letter_origin[1, 0]

    # Draw every letter
    for i, l in enumerate(word):
        if i > 0:
            letter_origin += rot_mat @ np.array(
                [[letter_width[i - 1] + spacing[i - 1]], [0]]
            )
        left, top, right, bottom = fnt.getbbox(l)
        img.paste(
            letter_patches[i],
            (int(letter_origin[0, 0]), int(letter_origin[1, 0] + letter_offset[i])),
            letter_patches[i],
        )

    if (
        not (0 <= original_x <= W - letter_width[0])
        or not (0 <= letter_origin[0, 0] <= W - letter_width[-1])
        or not (0 <= original_y <= H - letter_height[0])
        or not (0 <= letter_origin[1, 0] <= H - letter_height[-1])
    ):
        raise ValueError(f"Text width is bigger than image. Failed on size:{size}")

    return img


def free_cache():
    r"""Reset caches used by `adaptive_get_max_width`"""
    global _font_angle_cache
    global _optimized_angle_width_cache
    _font_angle_cache.clear()
    _optimized_angle_width_cache.clear()


def adaptive_get_max_width(
    word: str, spacing: int, rot: list[int] | frozenset[int], font: str, size: int
) -> int:
    r"""Get the max width for `word` with `spacing` space between each letters,
    with the font `font` in fontsize `size`. Max width is pooled from maximizing
    width rotation for each letter, the rotations being pooled from `rot`.
    The function relies on one cache that stores for each set of rotation a dict
    that maps the tuple `(letter, font, size)` to its maximum width, and another
    cache that stores, for every tuple of `(letter, font, size)`, the radius and
    angle of the diagonal of the box of the letter."""

    global _font_angle_cache
    global _optimized_angle_width_cache
    if not isinstance(rot, frozenset):
        rot_set = frozenset(rot)
    else:
        rot_set = rot
    cached_widths = _optimized_angle_width_cache.get(rot_set, {})
    fnt = ImageFont.truetype(get_fonts_dir() / f"{font}.ttf", size)
    width = 0
    for letter in word:
        key = (letter, font, size)
        if key in cached_widths:
            maximized_width = cached_widths[key]
        else:
            if key in _font_angle_cache:
                r, theta = _font_angle_cache[key]
            else:
                left, top, right, bottom = fnt.getbbox(letter)
                letter_width = right - left
                letter_height = bottom - top
                r = 2 * np.ceil(np.sqrt(letter_width**2 + letter_height**2) / 2)
                theta = np.arccos(letter_width / r)
                _font_angle_cache[key] = (r, theta)
            angles = np.array([theta + (angle / 180 * np.pi) for angle in rot])
            maximized_width = np.ceil(
                r * np.max(np.abs(np.stack((np.cos(angles), np.cos(np.pi - angles)))))
            )
            cached_widths[key] = maximized_width
        width += maximized_width
    width += (len(word) - 1) * spacing
    return width


def reworked_max_font_size(
    words: Sequence[str], rot: list[int], image_width: int = 224
) -> dict[str, int]:
    max_spacing = 3
    max_length = np.max([len(word) for word in words])
    candidates = [word for word in words if len(word) >= max_length - 3]
    real_candidates = []
    real_candidates.extend([word.lower() for word in candidates])
    real_candidates.extend([word.upper() for word in candidates])
    real_candidates.extend([word.title() for word in candidates])

    font_sizes = {}
    all_fonts = get_all_fonts()
    rot_set = frozenset(rot)

    for font in all_fonts:
        top_font_size = 1
        low_font_size = 1
        top_is_ok = True
        while top_is_ok:
            low_font_size = top_font_size
            top_font_size *= 2
            max_width = np.max(
                [
                    adaptive_get_max_width(
                        word, max_spacing, rot_set, font, top_font_size
                    )
                    for word in real_candidates
                ]
            )
            top_is_ok = max_width <= image_width
        while low_font_size != top_font_size:
            current_font_size = (top_font_size + low_font_size) // 2
            if current_font_size == low_font_size:
                break
            max_width = np.max(
                [
                    adaptive_get_max_width(
                        word, max_spacing, rot_set, font, current_font_size
                    )
                    for word in real_candidates
                ]
            )
            if max_width <= image_width:
                low_font_size = current_font_size
            else:
                top_font_size = current_font_size
        font_sizes[font] = low_font_size
    return font_sizes


class ByFontArgs(TypedDict):
    r"""TypedDict containing values required to images of samples.
    Argument values for size are sorted by font. It is expected that all fonts have `"num_sizes"` sizes available.
    `"query"` contains the query passed to the dataframes to get the values contained in the dictionnary.
    """

    query: str
    font2sizes: dict[str, list[int]]
    global_rotations: list[int]
    line_rot: list[int]
    letter_rotations: list[int]
    spaces: list[int]
    num_sizes: int


def create_gen_arg_dict(
    path: Path,
    words: Sequence[str],
    query: str | None = None,
) -> ByFontArgs:
    r"""Generate a dictionnary of the arguments used to generate the grapheme dataset and save it."""
    train_path = path / "train"
    train_path.mkdir(exist_ok=True, parents=True)
    rotations = [0]  # old rotations : [-15, -10, -5, 0, 5, 10, 15]
    fonts2maxsize = reworked_max_font_size(words, rot=rotations)
    line_rot = [0]
    spaces = [
        0,
        1,
        2,
        3,
    ]
    fonts2sizes = {}
    for font, max_size in fonts2maxsize.items():
        fonts2sizes[font] = [int(2 * max_size / 3), int(5 * max_size / 6), max_size]
    num_sizes = 3
    dataset_gen_dict = ByFontArgs(
        {
            "query": query if query is not None else "",
            "font2sizes": fonts2sizes,
            "global_rotations": list(set(rotations)),
            "line_rot": list(set(line_rot)),
            "letter_rotations": list(set(rotations)),
            "spaces": list(set(spaces)),
            "num_sizes": num_sizes,
        }
    )
    gen_args_path = train_path / "gen_args.json"
    with gen_args_path.open("w") as f:
        json.dump(dataset_gen_dict, f, indent=4)
    return dataset_gen_dict


def get_gen_arg_dict(path) -> ByFontArgs:
    r"""Return the arguments used to generate the dataset stored in `path / "train"` directory"""
    train_gen_args_path = path / "train" / "gen_args.json"
    with train_gen_args_path.open("r") as f:
        gen_arg_dict = json.load(f)
    return ByFontArgs(gen_arg_dict)
