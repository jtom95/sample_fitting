from typing import Union, List, Tuple, Optional, Callable
from copy import deepcopy
import logging
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from my_packages.constants import DistanceUnits
from ._gaussian_regression_plotting import GaussianRegressionPlotterMixin
from .kernel_manager import KernelManager   

from ..abstract_sample_model import AbstractSampleModel
from ..surrogate_model import SurrogateModel


@dataclass
class GaussianRegressionModelConfig:
    normalization_method: str = "standard"
    max_range: float = None
    units: DistanceUnits = DistanceUnits.mm
    n_restarts_optimizer: int = 10
    alpha: float = 1e-10
    kernel_base_function: Callable = None
    kernel_kwargs: dict = None
    kernel_non_normalize_hyperparams: dict = None
    rescale_kernel_constant: bool = False

    def __post_init__(self):
        if self.kernel_kwargs is None:
            self.kernel_kwargs = {}
        if self.kernel_non_normalize_hyperparams is None:
            self.kernel_non_normalize_hyperparams = {}
        if isinstance(self.units, str):
            self.units = DistanceUnits[self.units]

class GaussianRegressionModel(AbstractSampleModel, GaussianRegressionPlotterMixin):  
    """
    A wrapper class for Gaussian Process Regression using scikit-learn's GaussianProcessRegressor.
    Provides methods for fitting, predicting, and evaluating the Gaussian Process model.
    """
    def __init__(
        self,
        configs: GaussianRegressionModelConfig = GaussianRegressionModelConfig(),
        **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        # if there are kwargs
        for key, value in kwargs.items():
            if hasattr(configs, key):
                setattr(configs, key, value)
            else:
                self.logger.warning(f"Unknown parameter {key} with value {value}: ignoring.")
        # run the post init method again
        configs.__post_init__()
        self.configs = configs    

        self._set_normalization_scaler(method=configs.normalization_method)
        self._set_label_scaler()

        if configs.kernel_base_function is None:
            self.kernelM = KernelManager()
        else:
            self.kernelM = KernelManager(
                kernel_function=configs.kernel_base_function,
                kernel_params=configs.kernel_kwargs,
                non_normalize_params=configs.kernel_non_normalize_hyperparams,
            )

    @property
    def units(self):
        return self.configs.units

    @property
    def n_restarts_optimizer(self):
        return self.configs.n_restarts_optimizer

    @n_restarts_optimizer.setter
    def n_restarts_optimizer(self, value):
        self.configs.n_restarts_optimizer = value

    @property
    def alpha(self):
        return self.configs.alpha
    @alpha.setter
    def alpha(self, value):
        self.configs.alpha = value

    @property
    def rescaled_alpha(self):
        if not hasattr(self, "label_scaler"):
            raise ValueError("The model has not been fitted yet.")
        return self.alpha * self.label_scale_**2

    @property
    def max_range(self):
        return self.configs.max_range
    @max_range.setter
    def max_range(self, value):
        self.configs.max_range = value

    @property
    def normalization_method(self):
        return self.configs.normalization_method
    @normalization_method.setter
    def normalization_method(self, value):
        self.configs.normalization_method = value
        self._set_normalization_scaler(value)

    @property
    def surrogate_model(self):
        return SurrogateModel(self)

    def update_configs_properties(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.configs, key, value)
        self.configs.__post_init__()
        return self

    def update_kernel(
        self, base_kernel: Callable = None, non_normalize_hyperparams: dict = None, **kwargs
    ) -> "GaussianRegressionModel":
        """
        Updates the kernel of the Gaussian Regression model.

        Args:
            base_kernel (Callable, optional): The base kernel function to use. Defaults to None.
            non_normalize_hyperparams (dict, optional): Non-normalized hyperparameters for the kernel. Defaults to None.
            **kwargs: Additional keyword arguments for the kernel function.

        Returns:
            GaussianRegressionModel: The updated Gaussian Regression model.

        """
        base_kernel = base_kernel or KernelManager.DEFAULT_KERNEL_FUNCTION
        non_normalize_hyperparams = (
            non_normalize_hyperparams or KernelManager.DEFAULT_NON_NORMALIZE_PARAMS.copy()
        )

        self.kernelM = KernelManager(
            kernel_function=base_kernel,
            kernel_params=kwargs,
            non_normalize_params=non_normalize_hyperparams,
        )
        return self

    def _set_normalization_scaler(self, method: str):
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
        }
        self.scaler = scalers.get(method)

    def _set_label_scaler(self):
        self.label_scaler = MinMaxScaler(feature_range=(-0.5, 0.5))

    @property
    def label_scale_(self):
        if hasattr(self, "label_scaler") and hasattr(self.label_scaler, "scale_"):
            if self.label_scaler.scale_ is not None:
                return self.label_scaler.scale_
        return 1.0

    def _preprocess_data(self, X: np.ndarray, fit=False) -> np.ndarray:
        X = X / self.units.value
        if self.scaler:
            if fit:
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)

    def initalize_gp(self, X, rescale_alpha=False, **kwargs):
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        self.X_scaled = self._preprocess_data(X, fit=True)
        self.kernelM.set_scaler(self.scaler)
        self.kernel = self.kernelM.make_kernel(
            normalize=True, max_range=self.max_range
            )

        if rescale_alpha:
            self.alpha = self.rescaled_alpha

        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            alpha=self.alpha,
            **kwargs,
        )
        return self

    def fit(self, X, y, rescale_alpha=False, **kwargs) -> "GaussianRegressionModel":
        """
        Fit the Gaussian Process model.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param y: Observed field values at these coordinates (shape [n_samples]).
        """

        # change the units of the spatial coordinates
        self.y_scaled = self.label_scaler.fit_transform(y.reshape(-1, 1))

        if self.configs.rescale_kernel_constant:
            self.kernelM.kernel_non_normalize_params["constant"] *= self.label_scale_

        self.initalize_gp(X, rescale_alpha=rescale_alpha, **kwargs)
        # X_scaled = self._preprocess_data(X, fit=True)
        # self.kernelM.set_scaler(self.scaler)
        # self.kernel = self.kernelM.make_kernel(
        #     normalize=True, max_range=self.max_range
        #     )
        # # self.kernel = self.make_kernel(normalize=True, max_range=max_range)

        # self.gp = GaussianProcessRegressor(
        #     kernel=self.kernel,
        #     n_restarts_optimizer=self.n_restarts_optimizer,
        #     alpha=self.alpha,
        #     **kwargs,
        # )
        self.gp.fit(self.X_scaled, self.y_scaled)
        return self

    def sample_prior_gp(self, X: np.ndarray, n_samples: int=5, random_state = None) -> np.ndarray:
        """
        Make predictions using the prior GP.
        :param X: 2D array of spatial coordinates for prediction (shape [n_samples, 2]).
        :return: Predicted field values at the given coordinates (shape [n_samples]).
        """
        self.initalize_gp(X)
        y_predicted = self.gp.sample_y(self.X_scaled, n_samples=n_samples, random_state=random_state)
        return y_predicted

    def sample_gp_candidates(self, X_test: np.ndarray, n_samples: int=5, random_state = None) -> np.ndarray:
        """
        Make predictions using the prior GP.
        :param X: 2D array of spatial coordinates for prediction (shape [n_samples, 2]).
        :return: Predicted field values at the given coordinates (shape [n_samples]).
        """
        if not hasattr(self, "X_scaled"):
            raise ValueError("The model has not been fitted yet.")

        X_scaled = self._preprocess_data(X_test, fit=False)
        y_predicted = self.gp.sample_y(X_scaled, n_samples=n_samples, random_state=random_state)
        y_predicted = self.label_scaler.inverse_transform(y_predicted)
        return y_predicted

    def predict_prior_gp(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the prior GP.
        :param X: 2D array of spatial coordinates for prediction (shape [n_samples, 2]).
        :param return_std: If True, returns the standard deviation of the predictions.
        :return: Predicted field values at the given coordinates (shape [n_samples]).
                 If return_std is True, also returns the standard deviation of the predictions.
        """
        self.initalize_gp(X)
        y_predicted = self.gp.predict(self.X_scaled, return_std=return_std)
        if return_std:
            y_predicted, std_devs = y_predicted
            return y_predicted, std_devs
        return y_predicted

    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the fitted model.
        :param X: 2D array of spatial coordinates for prediction (shape [n_samples, 2]).
        :param return_std: If True, returns the standard deviation of the predictions.
        :return: Predicted field values at the given coordinates (shape [n_samples]).
                 If return_std is True, also returns the standard deviation of the predictions.
        """
        X_scaled = self._preprocess_data(X, fit=False)
        y_predicted = self.gp.predict(X_scaled, return_std=return_std)
        if return_std:
            y_predicted, std_devs = y_predicted
            return self.label_scaler.inverse_transform(y_predicted.reshape(-1, 1)), std_devs / self.label_scale_
        return self.label_scaler.inverse_transform(y_predicted.reshape(-1, 1))

    def extract_theta(self):
        return self.gp.kernel_.theta

    def set_kernel_theta(self, params: dict, K0: Kernel):
        self.kernel = K0
        self.kernel = self.kernel.clone_with_theta(params)
        return self.kernel

    def fit_from_theta(self, X, y, theta: dict, K0: Kernel, rescale_alpha: bool = False, **kwargs)-> "GaussianRegressionModel":
        """
        Fit the Gaussian Process model.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param y: Observed field values at these coordinates (shape [n_samples]).
        """
        # change the units of the spatial coordinates
        X_scaled = self._preprocess_data(X, fit=True)
        y_scaled = self.label_scaler.fit_transform(y.reshape(-1, 1))
        self.kernel = self.set_kernel_theta(theta, K0)

        alpha = self.rescaled_alpha if rescale_alpha else self.alpha

        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            alpha=alpha,
            **kwargs
        )

        self.gp.fit(X_scaled, y_scaled)
        return self

    def prediction_std(self, X) -> np.ndarray:
        """
        Make predictions using the fitted model.
        :param X: 2D array of spatial coordinates for prediction.
        :return: Predicted field values at these coordinates and the std of the model.
        """
        # change the units of the spatial coordinates
        X = self._preprocess_data(X, fit=False)
        _, std = self.gp.predict(X, return_std=True)[1]

        std = std / self.label_scale_
        return std

    def get_kernel_params(self) -> dict:
        """
        Retrieve the kernel parameters (length scales and bounds).
        :return: Dictionary containing the length scales and bounds.
        """
        length_scales = []
        length_scale_bounds = []

        constant_value = None
        for param_name, param_value in self.gp.kernel_.get_params().items():
            if 'length_scale' in param_name:
                if 'bounds' not in param_name:
                    length_scales.append(param_value)
                else:
                    length_scale_bounds.append(param_value)

            # include the ConstantKernel parameter
            if 'constant_value' in param_name and 'bounds' not in param_name:
                if self.configs.rescale_kernel_constant and hasattr(self, "label_scaler") and hasattr(self.label_scaler, "scale_"):
                    constant_value = param_value / self.label_scale_  
                else:
                    constant_value = param_value

        if self.scaler:
            length_scales = np.asarray(length_scales) * np.mean(self.scaler.scale_) if length_scales is not None else None
            if not any([isinstance(b, str) or b is None for b in length_scale_bounds]) and not length_scale_bounds is None:
                length_scale_bounds = np.asarray(length_scale_bounds) * np.mean(self.scaler.scale_)

        return {
            'constant_value': constant_value,   
            'length_scales': length_scales,
            'length_scale_bounds': length_scale_bounds,
        }

    @property
    def kernel_constant(self):
        if hasattr(self, "gp") and self.gp is not None:
            if hasattr(self.gp, "kernel_"):
                return self.get_kernel_params()["constant_value"]
        C = self.kernelM.kernel_non_normalize_params.get("constant")
        if self.configs.rescale_kernel_constant and hasattr(self, "label_scaler") and hasattr(self.label_scaler, "scale_"):
            return C / self.label_scale_
        return C

    @property
    def kernel_length_scale(self):
        if hasattr(self, "gp") and self.gp is not None:
            if hasattr(self.gp, "kernel_"):
                return self.get_kernel_params()["length_scales"]
        return self.kernelM.kernel_params.get("length_scale")


class OtherGaussianRegressionModel(GaussianRegressionModel):
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
