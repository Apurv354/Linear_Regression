import unittest
import pandas as pd
from regression.__main__ import Regression


class RegressionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = Regression()
        self.reg.model_name = "Test_Linear_Regression_Model"

    def test_predictions(self):
        # Set path to data
        self.reg.train_data = pd.read_csv("../data/regression_test.csv", index_col="ID")
        test_data = pd.read_csv("../data/regression_test.csv", index_col="ID")
        model = self.reg.predict(["height", "weight", "age"], ["BicepC"], False)
        r_sq = model.score(test_data.loc[:, ["height", "weight", "age"]], test_data.loc[:, ["BicepC"]])
        self.assertAlmostEqual(r_sq, 0.5, msg="Bad Fit", delta=0.1)


if __name__ == "__main__":
    unittest.main()
