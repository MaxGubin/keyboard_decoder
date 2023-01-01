import torch
import keyboard_simulator
import pandas as pd

class KeyboardTextDataset(torch.utils.data.Dataset):
    def __init__(self):
        super.__init__(self)
        df = pd.read_csv("imdb_dataset.csv")
        df = df.drop_duplicates()
        self.docs = df['review']
    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        in_text = self.docs[idx]
        encoded_text = keyboard_simulator.keyboard_encode_coordinates(in_text)[1]