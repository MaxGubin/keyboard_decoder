import unittest
import keyboard_simulator
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch.utils.data as data

class KeyboardSimulatorTest(unittest.TestCase):
    def test_EncodeString(self):
        sample_strings = ["abc", "I'm coming home.", "I'll be back"]
        simulator = keyboard_simulator.KeyboardDataset(sample_strings)
        self.assertEqual(len(simulator), len(sample_strings))
        for s,e in zip(sample_strings, simulator):
            self.assertEqual(len(s[0]), len(e))

    def test_RandomString(self):
        NUM_SAMPLES = 100
        simulator = keyboard_simulator.RandomKeyboardDataset(dataset_size=NUM_SAMPLES)
        self.assertEqual(NUM_SAMPLES, len(simulator))
        for x,y in enumerate(simulator):
            print(y)
        self.assertEqual(x, NUM_SAMPLES-1)

    def test_PytorchDataset(self):
        df = pd.read_csv("imdb_dataset.csv")
        df = df.drop_duplicates()
        docs = df['review']
        X_train, _ = train_test_split(docs, test_size = 20, random_state=0)
        dataset = keyboard_simulator.KeyboardDataset(X_train)
        dataloader = data.DataLoader(dataset) 
        for x,i in  enumerate(dataloader):
            print(x,i)



if __name__ == '__main__':
    unittest.main()