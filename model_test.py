import unittest
import model


class ModelTest(unittest.TestCase):
    def test_ModelCreate(self):
        _model = model.BuildModel()
        self.assertIsNotNone(_model)


if __name__ == '__main__':
    unittest.main()