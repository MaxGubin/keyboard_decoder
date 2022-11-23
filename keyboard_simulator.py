import collections
import string
from collections.abc import Sequence
import re
import random

import torch.utils.data as data

CGPoint = collections.namedtuple("CGPoint", ['x', 'y'])

QWERTY_MAP = {
    ' ': CGPoint(x=0.50, y=0.063),

    'q': CGPoint(x=0.05, y=0.125),
    'w': CGPoint(x=0.15, y=0.125),
    'e': CGPoint(x=0.25, y=0.125),
    'r': CGPoint(x=0.35, y=0.125),
    't': CGPoint(x=0.45, y=0.125),
    'y': CGPoint(x=0.55, y=0.125),
    'u': CGPoint(x=0.65, y=0.125),
    'i': CGPoint(x=0.75, y=0.125),
    'o': CGPoint(x=0.85, y=0.125),
    'p': CGPoint(x=0.95, y=0.125),
        
    'a': CGPoint(x=0.10, y=0.375),
    's': CGPoint(x=0.20, y=0.375),
    'd': CGPoint(x=0.30, y=0.375),
    'f': CGPoint(x=0.40, y=0.375),
    'g': CGPoint(x=0.50, y=0.375),
    'h': CGPoint(x=0.60, y=0.375),
    'j': CGPoint(x=0.70, y=0.375),
    'k': CGPoint(x=0.80, y=0.375),
    'l': CGPoint(x=0.90, y=0.375),
        
    'z': CGPoint(x=0.20, y=0.625),
    'x': CGPoint(x=0.30, y=0.625),
    'c': CGPoint(x=0.40, y=0.625),
    'v': CGPoint(x=0.50, y=0.625),
    'b': CGPoint(x=0.60, y=0.625),
    'n': CGPoint(x=0.70, y=0.625),
    'm': CGPoint(x=0.80, y=0.625),

}

NUM_CLASSES = len(QWERTY_MAP.keys())
REPLACE_WITH_SPACE = '[^a-z]'

def normalize_string(text:str)->str:
    return re.sub(REPLACE_WITH_SPACE, ' ', text.lower())

def keyboard_encode_coordinates(text: str)->Sequence[CGPoint]:
    result=[]
    for c in text.lower():
        if c in QWERTY_MAP:
            result.append(QWERTY_MAP[c])
    return result


class KeyboardDataset(data.Dataset):
    def __init__(self, text_set:Sequence[str]):
        self.text_set = text_set

    def __len__(self):
        return len(self.text_set)
    
    def __getitem__(self, idx)->Sequence[CGPoint]:
        test_string = normalize_string(self.text_set[idx])
        encoded = keyboard_encode_coordinates(test_string)
        return encoded

def generate_random_char():
    rand_index = random.randint(0, ord('z') - ord('a') + 1)
    return ' ' if 0 == rand_index else chr(ord('a') + rand_index-1)

def generate_random_string():
    random_len = random.randint(5, 15)
    return "".join(( generate_random_char() for _ in range(random_len))) 

class RandomKeyboardDataset(KeyboardDataset):
    def __init__(self, dataset_size):
        super.__init__([generate_random_string() for _ in range(dataset_size)])