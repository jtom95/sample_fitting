import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GaussianRegressionModel:
    def __init__(self):
        # Define the kernel to be used in the Gaussian Process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    def fit(self, X, y):
        """
        Fit the Gaussian Process model.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param y: Observed field values at these coordinates (shape [n_samples]).
        """
        self.gp.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the fitted model.
        :param X: 2D array of spatial coordinates for prediction.
        :return: Predicted field values at these coordinates and the std of the model.
        """
        return self.gp.predict(X, return_std=True)


# Example usage
if __name__ == "__main__":
    # Sample data (replace with real data)
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Sample points
    y = np.array([2.3, 3.5, 5.6])  # Observed field values

    model = GaussianRegressionModel()
    model.fit(X, y)

    # Predict at new points
    X_new = np.array([[2, 3], [4, 5]])
    predictions, std_devs = model.predict(X_new)
    print("Predictions:", predictions)
    print("Standard Deviations:", std_devs)
