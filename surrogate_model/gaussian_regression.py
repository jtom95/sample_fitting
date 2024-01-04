from typing import Union, List, Tuple, Optional
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from my_packages.constants import (
    FieldComponents,
    FieldTypes,
    Orientations,
    ProbeTypes,
    DistanceUnits,
)

from .custom_scikit_kernel import MyRBF, StepKernel


from my_packages.EM_fields.scans import Scan
from my_packages.classes.aux_classes import Grid


class GaussianRegressionModel:
    def __init__(
        self,
        length_scales: Union[float, List[float]] = 1,
        length_scale_bounds: Tuple[float, float] = (1e-2, 1e2),
        normalize: str = "standard",
        max_range: float = None,
        units: DistanceUnits = DistanceUnits.mm,
        n_restarts_optimizer: int = 10,
        alpha: float = 1e-10
    ):
        if isinstance(units, str):
            units = DistanceUnits[units]

        self.n_restarts_optimizer = n_restarts_optimizer
        self.length_scales = length_scales
        self.length_scale_bounds = length_scale_bounds
        self.max_range = max_range
        self.alpha = alpha
        self.units = units
        self.set_normalization_scaler(method=normalize)

    def make_kernel(
        self, normalize=True, 
        length_scale=None, 
        length_scale_bounds=None, 
        max_range=None
        ):
        if length_scale is None:
            length_scale = self.length_scales
        if length_scale_bounds is None:
            length_scale_bounds = self.length_scale_bounds
        if max_range is None:
            max_range = self.max_range
        
        if normalize:
            length_scale, length_scale_bounds, max_range = self.normalize_input_hyperparams(
                length_scale, length_scale_bounds, max_range
            )

        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=1.5,
        )
        if max_range is not None:
            kernel *= StepKernel(max_range=max_range)
        return kernel

    def set_normalization_scaler(self, method="standard"):
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        return self

    @property
    def K(self):
        return self.gp.kernel_

    # def distances(self, X, Y=None):
    #     return CustomRBFKernel._calculate_distances(X, Y)

    def normalize_input_hyperparams(self, length_scale, length_scale_bounds, max_range):
        # normalize length scale
        length_scale /= self.scaler.scale_[0]
        length_scale_bounds = tuple(np.asarray(length_scale_bounds) / self.scaler.scale_[0])
        # normalize max range
        if max_range is not None:
            max_range /= self.scaler.scale_[0]
        return length_scale, length_scale_bounds, max_range

    def fit(self, X, y, n_restarts_optimizer=None, alpha=None, max_range=None, **kwargs):
        """
        Fit the Gaussian Process model.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param y: Observed field values at these coordinates (shape [n_samples]).
        """
        if n_restarts_optimizer is None:
            n_restarts_optimizer = self.n_restarts_optimizer
        if alpha is None:
            alpha = self.alpha

        # change the units of the spatial coordinates
        X = X / self.units.value
        X_scaled = self.scaler.fit_transform(X)
        
        self.kernel = self.make_kernel(normalize=True, max_range=max_range)
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
            **kwargs,
        )
        self.gp.fit(X_scaled, y)

    def extract_theta(self):
        return self.gp.kernel_.theta

    def set_kernel_theta(self, params: dict, normalize_bounds=False, max_range=None):
        self.kernel = self.make_kernel(normalize=normalize_bounds, max_range=max_range)
        self.kernel = self.kernel.clone_with_theta(params)
        return self.kernel

    def fit_from_theta(self, X, y, theta: dict, n_restarts_optimizer=None, alpha=None, max_range=None, **kwargs):
        """
        Fit the Gaussian Process model.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param y: Observed field values at these coordinates (shape [n_samples]).
        """
        if n_restarts_optimizer is None:
            n_restarts_optimizer = self.n_restarts_optimizer
        if alpha is None:
            alpha = self.alpha
        # change the units of the spatial coordinates
        X = X / self.units.value
        X_scaled = self.scaler.fit_transform(X)
        self.kernel = self.set_kernel_theta(theta, max_range=max_range, normalize_bounds=True)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
            **kwargs,
        )
        self.gp.fit(X_scaled, y)

    def score(self, X, y):
        """
        Score the Gaussian Process model.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param y: Observed field values at these coordinates (shape [n_samples]).
        """
        # change the units of the spatial coordinates
        X = X / self.units.value
        X_scaled = self.scaler.transform(X)
        return self.gp.score(X_scaled, y)

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the fitted model.
        :param X: 2D array of spatial coordinates for prediction.
        :return: Predicted field values at these coordinates and the std of the model.
        """
        # change the units of the spatial coordinates
        X = X / self.units.value
        X = self.scaler.transform(X)
        return self.gp.predict(X, return_std=True)

    def get_final_length_scales(self):
        """
        Retrieve the final length scales used in the Gaussian Process model.
        :return: Length scales as a numpy array.
        """
        if self.gp.kernel_ is not None:
            for hyp in self.gp.kernel_.get_params():
                if "length_scale" in hyp and "bounds" not in hyp:
                    length_scale = deepcopy(self.gp.kernel_.get_params()[hyp])
                if "length_scale_bounds" in hyp:
                    length_scale_bounds = deepcopy(self.gp.kernel_.get_params()[hyp])
            
            # denormalize length scale
            length_scale *= self.scaler.scale_[0]
            length_scale_bounds = tuple(np.asarray(length_scale_bounds) * self.scaler.scale_[0])
            
        return length_scale, length_scale_bounds

    def is_data_stationary(self, X, y, num_segments=4, threshold=0.05) -> Tuple[bool, bool]:
        """
        Check if the data is stationary, in terms of mean and variance.
        :param X: Spatial coordinates (2D array).
        :param y: Observed field values (1D array).
        :param num_segments: Number of segments to divide the data into.
        :param threshold: Threshold for relative difference in mean/variance to consider data as non-stationary.
        :return: Tuple of booleans indicating whether the data is stationary in mean and variance.
        """
        # Divide the data into segments
        segment_length = len(y) // num_segments
        means = []
        variances = []

        for i in range(num_segments):
            segment = y[i * segment_length : (i + 1) * segment_length]
            means.append(np.mean(segment))
            variances.append(np.var(segment))

        # Check for consistency in mean and variance
        mean_change = max(means) - min(means)
        var_change = max(variances) - min(variances)
        mean_reference = np.mean(means)
        var_reference = np.mean(variances)

        is_mean_stationary = (mean_change / mean_reference) < threshold
        is_var_stationary = (var_change / var_reference) < threshold

        return is_mean_stationary, is_var_stationary

    def predict_with_K(self, X_new, K: Kernel = None):
        """
        Make predictions using the fitted model and a custom kernel matrix.
        :param X_new: 2D array of spatial coordinates for prediction.
        :return: Predicted field values at these coordinates and the std of the model.
        """
        # set the kernel matrix to be used
        if K is None:
            K = self.K
        # Ensure X_new is correctly scaled
        X_new_scaled = X_new / self.units.value
        X_new_scaled = self.scaler.transform(X_new_scaled)

        # Get the kernel matrix for the training data with noise term
        K_train = K(self.gp.X_train_)
        K_train += np.eye(K_train.shape[0]) * self.gp.alpha  # Adding noise term

        # Kernel matrix between new data points and training data
        K_new = K(X_new_scaled, self.gp.X_train_)

        # Kernel matrix for new data points
        K_new_new = K(X_new_scaled)

        # Ensure numerical stability in matrix inversion using Cholesky decomposition
        L = np.linalg.cholesky(K_train)
        L_inv = np.linalg.inv(L)
        K_inv = L_inv.T @ L_inv

        # Compute the mean
        mean = K_new @ K_inv @ self.gp.y_train_

        # Compute the variance
        v = L_inv @ K_new.T
        variance = K_new_new - v.T @ v

        # Return the mean and standard deviation
        return mean, np.sqrt(np.diag(variance))

    def predict_on_grid(self, grid: Grid, frequency: float = None, **kwargs) -> Tuple[Scan, Scan]:
        if frequency is None:
            frequency = -1

        shape2d = (grid.shape[1], grid.shape[2])
        points = grid.create_position_matrix()
        points2d = points[:, 0:2]
        labels, std_devs = self.predict(points2d)
        labels = labels.reshape(shape2d)
        std_devs = std_devs.reshape(shape2d)
        labels_scan = Scan(labels, grid=grid, freq=frequency, **kwargs)
        std_scan = Scan(std_devs, grid=grid, freq=frequency, **kwargs)
        return labels_scan, std_scan

    @staticmethod
    def compare_scan(
        scan: Scan,
        fitting_points: np.ndarray,
        fitting_labels: np.ndarray,
        std_scan: Optional[Scan] = None,
        ax: Optional[np.ndarray] = None,
        units=None,
        **kwargs
    ):
        # overall max between fitting labels and labels_scan
        max_value = np.max([np.max(fitting_labels), np.max(scan.v)])
        min_value = np.min([np.min(fitting_labels), np.min(scan.v)])

        if ax is None:
            n_axes = 3 if std_scan is not None else 2
            def_height = 3 if std_scan is not None else 5
            fig, ax = plt.subplots(1, n_axes, figsize=(10, def_height), constrained_layout=True)
        else:
            fig = ax[0].get_figure()

        ax[0].scatter(
            fitting_points[:, 0],
            fitting_points[:, 1],
            c=fitting_labels,
            vmin=min_value,
            vmax=max_value,
            cmap="jet",
        )
        ax[0].set_xlabel("x (mm)")
        ax[0].set_ylabel("y (mm)")

        # transfrom to mm
        ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(x * 1e3)))
        ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(x * 1e3)))

        ax[0].set_title("Original Points")

        scan.plot(ax=ax[1], vmin=min_value, vmax=max_value, units=units, **kwargs)
        ax[1].set_title("Predicted Points")
        if std_scan is not None:
            std_scan.plot(ax=ax[2], units="std", **kwargs)
            ax[2].set_title("Standard Deviation")
        return fig, ax


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
