import collections
import string
from collections.abc import Sequence
import re
import random

import torch.utils.data as data

CGPoint = collections.namedtuple("CGPoint", ['x', 'y'])

# We assume that all keys have the same size and adding noise bigger than
# this size introduces typing outside of the key.
X_PRECISION=0.05
Y_PRECISION=0.125

QWERTY_MAP = {
    ' ': (0, CGPoint(x=0.50, y=0.875)),

    'q': (1, CGPoint(x=0.05, y=0.125)),
    'w': (2, CGPoint(x=0.15, y=0.125)),
    'e': (3, CGPoint(x=0.25, y=0.125)),
    'r': (4, CGPoint(x=0.35, y=0.125)),
    't': (5, CGPoint(x=0.45, y=0.125)),
    'y': (6, CGPoint(x=0.55, y=0.125)),
    'u': (7, CGPoint(x=0.65, y=0.125)),
    'i': (8, CGPoint(x=0.75, y=0.125)),
    'o': (9, CGPoint(x=0.85, y=0.125)),
    'p': (10, CGPoint(x=0.95, y=0.125)),

    'a': (11, CGPoint(x=0.10, y=0.375)),
    's': (12, CGPoint(x=0.20, y=0.375)),
    'd': (13, CGPoint(x=0.30, y=0.375)),
    'f': (14, CGPoint(x=0.40, y=0.375)),
    'g': (15, CGPoint(x=0.50, y=0.375)),
    'h': (16, CGPoint(x=0.60, y=0.375)),
    'j': (17, CGPoint(x=0.70, y=0.375)),
    'k': (18, CGPoint(x=0.80, y=0.375)),
    'l': (19, CGPoint(x=0.90, y=0.375)),

    'z': (20, CGPoint(x=0.20, y=0.625)),
    'x': (21, CGPoint(x=0.30, y=0.625)),
    'c': (22, CGPoint(x=0.40, y=0.625)),
    'v': (23, CGPoint(x=0.50, y=0.625)),
    'b': (24, CGPoint(x=0.60, y=0.625)),
    'n': (25, CGPoint(x=0.70, y=0.625)),
    'm': (26, CGPoint(x=0.80, y=0.625)),

    # PAD value
    '#': (27, CGPoint(x=-1, y=-1))
}

NUM_CLASSES = len(QWERTY_MAP.keys())
PAD_ID = 27
REPLACE_WITH_SPACE = '[^a-z]'


def normalize_string(text: str) -> str:
    return re.sub(REPLACE_WITH_SPACE, ' ', text.lower())


def keyboard_encode_coordinates(text: str) -> tuple[Sequence[int], Sequence[CGPoint]]:
    result_points = []
    result_codes = []
    for c in text.lower():
        if c in QWERTY_MAP:
            code, point = QWERTY_MAP[c]
            result_codes.append(code)
            result_points.append(point)
    return result_points, result_codes


def random_batch_sample(batch_size):
    random_string = "".join((generate_random_char()
                            for _ in range(batch_size)))
    return keyboard_encode_coordinates(random_string)


class KeyboardDataset(data.Dataset):
    def __init__(self, text_set: Sequence[str]):
        self.text_set = text_set

    def __len__(self):
        return len(self.text_set)

    def __getitem__(self, idx) -> Sequence[CGPoint]:
        test_string = normalize_string(self.text_set[idx])
        encoded = keyboard_encode_coordinates(test_string)
        return encoded


def generate_random_char():
    rand_index = random.randint(0, ord('z') - ord('a') + 1)
    return ' ' if 0 == rand_index else chr(ord('a') + rand_index-1)


def generate_random_string():
    random_len = random.randint(5, 15)
    return "".join((generate_random_char() for _ in range(random_len)))


class RandomKeyboardDataset(KeyboardDataset):
    def __init__(self, dataset_size):
        super().__init__([generate_random_string()
                          for _ in range(dataset_size)])
