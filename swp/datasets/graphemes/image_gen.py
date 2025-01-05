from math import ceil, sqrt
from string import ascii_letters
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_FONTS = ["brushscriptstd", "Pacifico-Regular"]
SANS_FONTS = ["Arial", "helvetica"]
SERIF_FONTS = ["Times_New_Roman", "Georgia"]
_font_spacing_cache = {}
_font_diameter_cache = {}


def get_all_fonts() -> list[str]:
    r"""Returns all fonts used for image generation"""
    return SERIF_FONTS + SANS_FONTS + SCRIPT_FONTS


def text_to_grapheme(
    word: str,
    fontname: str,
    W: int = 224,
    H: int = 224,
    size: int = 20,
    spacing: list[int] = [],
    angles: list[int] = [],
    case: list[bool] = [],
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

    if len(spacing) != len(word):
        spacing = spacing[: (len(word) - 1)]
        spacing += [0 for _ in range(len(word) - len(spacing))]
    if len(angles) != len(word):
        angles = angles[: (len(word))]
        angles += [0 for _ in range(len(word) - len(angles))]
    if len(case) != len(word):
        case = case[: (len(word))]
        case += [False for _ in range(len(word) - len(case))]
    word = "".join(
        [word[i].upper() if case[i] else word[i].lower() for i in range(len(word))]
    )

    img = Image.new("RGB", (W, H), color="white")
    fnt = ImageFont.truetype(fontname + ".ttf", size)
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
        or not (0 <= letter_origin[0, 0] <= H - letter_height[-1])
    ):
        raise ValueError(f"Text width is bigger than image. Failed on size:{size}")

    return img


def get_spacing(font: str, size: int) -> int:
    r"""Get the number of pixels corresponding to half a space in font `font` and size `size`, rounded down.

    Uses caching for faster results over several calls."""
    global _font_spacing_cache
    key = (font, size)
    if key in _font_spacing_cache:
        spacing = _font_spacing_cache[key]
    else:
        fnt = ImageFont.truetype(font + ".ttf", size)
        left, _, right, _ = fnt.getbbox(" ")
        spacing = int((right - left) / 2)
        _font_spacing_cache[key] = spacing
    return spacing


def get_max_char_diameter(font: str, size: int) -> int:
    r"""Get the number of pixels corresponding to the biggest diameter of all
    letters in font `font` and size `size`, rounded up. Iterates over both lower
    and upper letters.

    Uses caching for faster results over several calls."""
    global _font_diameter_cache
    key = (font, size)
    if key in _font_diameter_cache:
        diameter = _font_diameter_cache[key]
    else:
        diameter = -1
        fnt = ImageFont.truetype(font + ".ttf", size)
        for letter in ascii_letters:
            left, top, right, bottom = fnt.getbbox(letter)
            width = right - left
            height = bottom - top
            letter_diam = 2 * ceil(sqrt(width**2 + height**2) / 2)
            if letter_diam > diameter:
                diameter = letter_diam
        _font_diameter_cache[key] = diameter
    return diameter


def free_cache():
    r"""Reset caches used by `get_spacing` and `get_max_char_diameter`"""
    global _font_spacing_cache
    global _font_diameter_cache
    _font_spacing_cache = {}
    _font_diameter_cache = {}


def get_max_font_size(
    word_len: int, font: str, image_width: int = 224
) -> tuple[int, int]:
    r"""Estimate the maximum font size usable to write a word of size `word_len`
    in font `font` in an image of width `image_width`, with any letter rotation
    and spaces of half a space between each letter.

    Returns the determined font size and its corresponding spacing.

    The value is made by finding the biggest letter of the font (in terms of diameter),
    and computing the max font size for a word constituted only of this character, with
    the width-maximizing rotation.
    """
    top_font_size = 1
    low_font_size = 1
    top_is_ok = True
    while top_is_ok:
        low_font_size = top_font_size
        top_font_size *= 2
        space = get_spacing(font, top_font_size)
        diameter = get_max_char_diameter(font, top_font_size)
        top_is_ok = (word_len * diameter + (word_len - 1) * space) <= image_width
    max_space = get_spacing(font, low_font_size)
    while low_font_size != top_font_size:
        current_font_size = (top_font_size + low_font_size) // 2
        if current_font_size == low_font_size:
            break
        space = get_spacing(font, current_font_size)
        diameter = get_max_char_diameter(font, current_font_size)
        if (word_len * diameter + (word_len - 1) * space) <= image_width:
            low_font_size = current_font_size
            max_space = space
        else:
            top_font_size = current_font_size
    return low_font_size, max_space


def get_dataset_max_font_size(word_dataset: Sequence[str]) -> tuple[int, int]:
    r"""Get the maximum font size and spaces usable over `word_dataset` so images can be
    generated with any rotations and spacing of returned spacing.
    """
    max_len = len(max(word_dataset, key=len))
    max_size = None
    max_space = None
    all_fonts = {*SERIF_FONTS, *SANS_FONTS, *SCRIPT_FONTS}
    for font in all_fonts:
        font_size, spacing = get_max_font_size(max_len, font)
        if max_size is None:
            max_size = font_size
        elif font_size < max_size:
            max_size = font_size
        if max_space is None:
            max_space = spacing
        elif spacing < max_space:
            max_space = spacing
    if max_size is None or max_space is None:
        raise ValueError("No words in the dataset")
    return max_size, max_space
