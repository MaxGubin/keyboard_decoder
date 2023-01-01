import unittest
import language_model


class LanguageModelTest(unittest.TestCase):
    def test_lstm_dataset(self):
        training_data, validation_data = language_model.prepare_lstm_dataset()
        self.assertIsNotNone(training_data)
        self.assertIsNotNone(validation_data)

if __name__ == '__main__':
    unittest.main()