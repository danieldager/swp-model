import functools
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import numpy as np
import torch
from g2p_en import G2p
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import BatchSampler, DataLoader, Sampler, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

SCRIPT_FONTS = ["brushscriptstd", "Pacifico-Regular"]
SANS_FONTS = ["Arial", "helvetica"]
SERIF_FONTS = ["Times_New_Roman", "Georgia"]
_font_spacing_cache = {}


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
):
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

    return np.array(img)


def text_to_grapheme_bool(
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
):
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

    fnt = ImageFont.truetype(fontname + ".ttf", size)
    letter_width = []
    letter_height = []
    letter_patches = []
    letter_offset = []

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

    for i, l in enumerate(word):
        if i > 0:
            letter_origin += rot_mat @ np.array(
                [[letter_width[i - 1] + spacing[i - 1]], [0]]
            )

    if (
        not (0 <= original_x <= W - letter_width[0])
        or not (0 <= letter_origin[0, 0] <= W - letter_width[-1])
        or not (0 <= original_y <= H - letter_height[0])
        or not (0 <= letter_origin[0, 0] <= H - letter_height[-1])
    ):
        return False

    return True


def get_spacing(font: str, size: int) -> int:
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


def free_cache():
    global _font_spacing_cache
    _font_spacing_cache = {}


def get_max_font_size(word, font):
    top_font_size = 1
    low_font_size = 1
    top_is_ok = True
    while top_is_ok:
        low_font_size = top_font_size
        top_font_size *= 2
        space = get_spacing(font, top_font_size)
        top_is_ok = text_to_grapheme_bool(
            word, font, spacing=[space for _ in word], size=top_font_size
        )
    max_space = get_spacing(font, low_font_size)
    while low_font_size != top_font_size:
        current_font_size = (top_font_size + low_font_size) // 2
        if current_font_size == low_font_size:
            break
        space = get_spacing(font, current_font_size)
        if text_to_grapheme_bool(
            word, font, spacing=[space for _ in word], size=current_font_size
        ):
            low_font_size = current_font_size
            max_space = space
        else:
            top_font_size = current_font_size
    return low_font_size, max_space


def get_dataset_max_font_size(word_dataset):
    max_len = -1
    longests = set()
    for word in word_dataset:
        if len(word) > max_len:
            max_len = len(word)
            longests = {word}
        elif len(word) == max_len:
            longests.add(word)
    max_size = None
    max_space = None
    all_fonts = {*SERIF_FONTS, *SANS_FONTS, *SCRIPT_FONTS}
    for long_word in longests:
        for font in all_fonts:
            font_size, spacing = get_max_font_size(long_word, font)
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


def create_dataset(path, words, images_per_word):
    rotations = [-15, -10, -5, 0, 5, 10, 15]
    max_size, max_space = get_dataset_max_font_size(words)
    all_fonts = SERIF_FONTS + SANS_FONTS + SCRIPT_FONTS
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
        for arg in images_args:
            array_image = text_to_grapheme(**arg)
            im = Image.fromarray(array_image)
            im_name = f"{word}_{arg["fontname"]}_{arg["size"]}"
            im_name = f"{im_name}_l{arg["line_angle"]}"
            im_name = f"{im_name}_letrot{"-".join(arg["angles"])}"
            im_name = f"{im_name}_sp{"-".join(arg["spacing"])}"
            if not arg["case"][0]:
                case_name = "lowers"
            elif arg["case"][-1]:
                case_name = "uppers"
            else:
                case_name = "title"
            im_name = f"{im_name}_{case_name}"
            im_name = f"{im_name}.jpg"
            im.save(path / im_name)


class RepetitionDataset(ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        token_to_id,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        g2p = G2p()
        length_to_pad = 10  # max_len + 5

        def to_phoneme(word):
            phonemes = g2p(word)
            phonemes.append("<STOP>")
            phonemes.extend(["<PAD>" for _ in range(length_to_pad - len(phonemes))])
            return torch.Tensor([token_to_id(phoneme) for phoneme in phonemes])

        super().__init__(
            root,
            transform,
            to_phoneme,
            loader,
            is_valid_file,
            allow_empty,
        )
        self.class_to_sample_id = {}
        for sample_id, class_id in enumerate(self.targets):
            self.class_to_sample_id.setdefault(self.classes[class_id], []).append(
                sample_id
            )


class WordRandomSampler(Sampler):
    def __init__(
        self,
        word_to_freq: dict[str, float],
        word_to_ids: dict[str, list[int]],
        num_images_per_word: int,
        num_samples: int,
        replacement: bool = True,
        generator=None,
    ) -> None:
        self.words = []
        freqs = []
        for word, freq in word_to_freq.items():
            self.words.append(word)
            freqs.append(freq)
        self.freqs = torch.Tensor(freqs)
        self.num_images_per_word = num_images_per_word
        self.word_to_ids = word_to_ids
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.freqs, self.num_samples, self.replacement, generator=self.generator
        )
        rand_image_ids = torch.randint(
            low=0,
            high=self.num_images_per_word,
            size=(self.num_samples,),
            generator=self.generator,
        )
        sample_ids = []
        for rand_word_id, image_id in zip(
            rand_tensor.tolist(), rand_image_ids.tolist()
        ):
            sample_id = self.word_to_ids[self.words[rand_word_id]][image_id]
            sample_ids.append(sample_id)
        yield from iter(sample_ids)

    def __len__(self) -> int:
        return self.num_samples


# TODO create word_to_frequency
# TODO create phoneme_to_id


# NOTE: This function needs to be checked, missing image transformations
def get_image_train_data(self):
    grapheme_tensors = self.text_to_grapheme(self.words, self.savepath)
    grapheme_dataset = TensorDataset(*grapheme_tensors)
    grapheme_dataloader = DataLoader(
        grapheme_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
    )

    return grapheme_dataloader
