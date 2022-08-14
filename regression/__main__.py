import argparse
from typing import Sequence
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class Regression:
    """
    Regression library helps you to test and train regression models
    """
    def __int__(self):
        self.train_data = 0
        self.model_name = ""

    def predict(self, feature_matrix, target_vector, viz) -> LinearRegression:
        """
        Creates a Linear Regression model and helps you to predict target vector
        :param feature_matrix: Set of features corresponding to an object
        :param target_vector: Feature that we want to predict
        :param viz: To visualize the results
        :return: If viz is True graph else trained model
        """
        model = LinearRegression(
            fit_intercept=True, normalize=False, copy_X=True, n_jobs=None
        )
        x = self.train_data.loc[:, feature_matrix]
        y = self.train_data.loc[:, target_vector]
        model.fit(x, y)
        # print("Parameters M: {}".format(model.coef_))
        # print("Parameters b: {}".format(model.intercept_))
        if viz:
            self.plot_regression_results(
                y.values,
                model.predict(x),
                "MSE={:.2f} cm".format(mean_squared_error(y, model.predict(x))),
                target_vector[0],
            )
        else:
            return model

    def plot_regression_results(self, y_true, y_pred, scores, body_part) -> plt:
        """
        Scatter plot of the predicted vs true targets
        :param y_true: Original target values
        :param y_pred: Predicted target values
        :param scores: Model metrics
        :param body_part: Name of Target value
        :return: Plotted graph
        """
        _, ax = plt.subplots(figsize=(8, 8))
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "--r",
            linewidth=2,
        )
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
        ax.set_xlim([y_true.min(), y_true.max()])
        ax.set_ylim([y_true.min(), y_true.max()])
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        extra = plt.Rectangle(
            (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
        )
        ax.legend([extra], [scores], loc="upper left")
        title = self.model_name + "\n Target: {}".format(body_part)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


def main(arguments: Sequence[str] = None):
    parser = argparse.ArgumentParser(description="Parameters for Linear Regression")
    parser.add_argument(
        "--data_filepath",
        metavar="file_path",
        type=str,
        help="File Path for your training data",
        default="data/regression_train.csv",
    )
    parser.add_argument(
        "--feature_matrix",
        metavar="feature_matrix",
        default=["height", "weight", "age"],
        type=str,
        help="Feature Matrix",
    )
    parser.add_argument(
        "--target_vector",
        metavar="target_vector",
        default=["BicepC"],
        type=str,
        help="target_vector",
    )
    parser.add_argument(
        "--visualization",
        metavar="visualization",
        default=True,
        type=bool,
        help="visualization"
    )
    args = parser.parse_args(arguments)
    reg = Regression()
    reg.train_data = pd.read_csv(args.data_filepath, index_col="ID")
    reg.model_name = "LinearRegression"
    reg.predict(args.feature_matrix, args.target_vector, args.visualization)


if __name__ == "__main__":
    main()
