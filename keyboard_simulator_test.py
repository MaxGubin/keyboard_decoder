import unittest
import keyboard_simulator

class KeyboardSimulatorTest(unittest.TestCase):
    def test_EncodeString(self):
        sample_strings = ["abc", "I'm coming home.", "I'll be back"]
        simulator = keyboard_simulator.KeyboardDataset(sample_strings)
        self.assertEqual(len(simulator), len(sample_strings))
        for s,e in zip(sample_strings, simulator):
            self.assertEqual(len(s), len(e))

    def test_RandomString(self):
        NUM_SAMPLES = 100
        simulator = keyboard_simulator.RandomKeyboardDataset(dataset_size=NUM_SAMPLES)
        self.assertEqual(NUM_SAMPLES, len(simulator))
        for x,y in enumerate(simulator):
            print(y)
        self.assertEqual(x, NUM_SAMPLES-1)

if __name__ == '__main__':
    unittest.main()