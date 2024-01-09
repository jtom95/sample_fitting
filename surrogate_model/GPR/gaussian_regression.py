from typing import Union, List, Tuple, Optional, Callable
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
from .kernel_manager import KernelManager   

from ..abstract_sample_model import AbstractSampleModel
from ..surrogate_model import SurrogateModel

from my_packages.EM_fields.scans import Scan
from my_packages.classes.aux_classes import Grid


class GaussianRegressionModel(AbstractSampleModel):  
    def __init__(
        self,
        normalization_method: str = "standard",
        max_range: float = None,
        units: DistanceUnits = DistanceUnits.mm,
        n_restarts_optimizer: int = 10,
        alpha: float = 1e-10,
        kernel_base_function: Callable = None,
        kernel_kwargs: dict = None,
        kernel_non_normalize_hyperparams: dict = None,
    ):
        if isinstance(units, str):
            units = DistanceUnits[units]

        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_range = max_range
        self.alpha = alpha
        self.units = units
        self._set_normalization_scaler(method=normalization_method)
        
        if kernel_base_function is None:
            self.kernelM = KernelManager()
        else:
            self.kernelM = KernelManager(
                kernel_function=kernel_base_function,
                kernel_params=kernel_kwargs,
                non_normalize_params=kernel_non_normalize_hyperparams,
            )
    
    @property
    def surrogate_model(self):
        return SurrogateModel(self)

    def update_kernel(self, base_kernel: Callable = None, non_normalize_hyperparams: bool = None, **kwargs)-> "GaussianRegressionModel":
        if non_normalize_hyperparams is None:
            if base_kernel is None:
                non_normalize_hyperparams = KernelManager.DEFAULT_NON_NORMALIZE_PARAMS.copy()
            else:
                non_normalize_hyperparams = {}
            
        if base_kernel is None:
            base_kernel = KernelManager.DEFAULT_KERNEL_FUNCTION
        
        kernel_base_function = base_kernel
        kernel_kwargs = kwargs
        non_normalize_hyperparams = non_normalize_hyperparams
        
        self.kernelM = KernelManager(
            kernel_function=kernel_base_function,
            kernel_params=kernel_kwargs,
            non_normalize_params=non_normalize_hyperparams,
        )
        return self        

    def _set_normalization_scaler(self, method="standard"):
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        return self

    def fit(self, X, y, n_restarts_optimizer=None, alpha=None, max_range=None, **kwargs) -> "GaussianRegressionModel":
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
        
        self.kernelM.set_scaler(self.scaler)
        self.kernel = self.kernelM.make_kernel(normalize=True, max_range=max_range)
        # self.kernel = self.make_kernel(normalize=True, max_range=max_range)
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
            **kwargs,
        )
        self.gp.fit(X_scaled, y)
        return self
        

    def extract_theta(self):
        return self.gp.kernel_.theta

    def set_kernel_theta(self, params: dict, K0: Kernel):
        self.kernel = K0
        self.kernel = self.kernel.clone_with_theta(params)
        return self.kernel

    def fit_from_theta(self, X, y, theta: dict, K0: Kernel, n_restarts_optimizer=None, alpha=None, max_range=None, **kwargs)-> "GaussianRegressionModel":
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
        self.kernel = self.set_kernel_theta(theta, K0)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
            **kwargs,
        )
        self.gp.fit(X_scaled, y)
        return self

    def predict_with_std(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the fitted model.
        :param X: 2D array of spatial coordinates for prediction.
        :return: Predicted field values at these coordinates and the std of the model.
        """
        # change the units of the spatial coordinates
        X = X / self.units.value
        X = self.scaler.transform(X)
        return self.gp.predict(X, return_std=True)

    def prediction_std(self, X) -> np.ndarray:
        """
        Make predictions using the fitted model.
        :param X: 2D array of spatial coordinates for prediction.
        :return: Predicted field values at these coordinates and the std of the model.
        """
        # change the units of the spatial coordinates
        X = X / self.units.value
        X = self.scaler.transform(X)
        return self.gp.predict(X, return_std=True)[1]
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the fitted model.
        :param X: 2D array of spatial coordinates for prediction.
        :return: Predicted field values at these coordinates and the std of the model.
        """
        # change the units of the spatial coordinates
        X = X / self.units.value
        X = self.scaler.transform(X)
        return self.gp.predict(X)

    def get_final_length_scales(self):
        """
        Retrieve the final length scales used in the Gaussian Process model.
        :return: Length scales as a numpy array.
        """
        length_scales = []
        if self.gp.kernel_ is not None:
            for hyp in self.gp.kernel_.get_params():
                if "length_scale" in hyp and "bounds" not in hyp:
                    length_scale = deepcopy(self.gp.kernel_.get_params()[hyp])
                    length_scales.append(length_scale)
                # if "length_scale_bounds" in hyp:
                #     length_scale_bounds = deepcopy(self.gp.kernel_.get_params()[hyp])
            
            # denormalize length scale
            length_scales = np.asarray(length_scales) * self.scaler.scale_[0]
            # length_scale_bounds = tuple(np.asarray(length_scale_bounds) * self.scaler.scale_[0])    
        return length_scales
    
    def get_length_scale_bounds(self):
        all_length_scale_bounds = []
        if self.gp.kernel_ is not None:
            for hyp in self.gp.kernel_.get_params():
                if "length_scale_bounds" in hyp:
                    length_scale_bounds = deepcopy(self.gp.kernel_.get_params()[hyp])
                    all_length_scale_bounds.append(np.asarray(length_scale_bounds))
            # denormalize length scale
            length_scale_bounds = np.asarray(all_length_scale_bounds) * self.scaler.scale_[0]
        return length_scale_bounds
    

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
