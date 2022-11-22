import unittest
import model


class ModelTest(unittest.TestCase):
    def test_ModelCreate(self):
        _model, _status = model.BuildModel()
        self.assertIsNotNone(_model)
        self.assertIsNotNone(_status)


if __name__ == '__main__':
    unittest.main()