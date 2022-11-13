import unittest
import keyboard_simulator

class KeyboardSimulatorTest(unittest.TestCase):
    def test_EncodeString(self):
        sample_strings = ["abc", "I'm coming home.", "I'll be back"]
        simulator = keyboard_simulator.KeyboardDataset(sample_strings)
        self.assertEqual(len(simulator), len(sample_strings))
        for s,e in zip(sample_strings, simulator):
            self.assertEqual(len(s), len(e))

if __name__ == '__main__':
    unittest.main()