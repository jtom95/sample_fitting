from typing import Callable, Union
import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .custom_scikit_kernel import StepKernel


class KernelManager:
    DEFAULT_KERNEL_FUNCTION = staticmethod(
        lambda length_scale, length_scale_bounds, nu: C(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=nu,
        )
    )
    DEFAULT_KERNEL_PARAMS = {
        "length_scale": 1.0,
        "length_scale_bounds": (1e-3, 1e3),
    }

    DEFAULT_NON_NORMALIZE_PARAMS = {
        "nu": 1.5,
    }

    def __init__(
        self,
        kernel_function: Callable = None,
        kernel_params: dict = None,
        non_normalize_params: dict = None,
    ):
        self.kernel_function = kernel_function or self.DEFAULT_KERNEL_FUNCTION
        self.kernel_normalize_params = kernel_params or self.DEFAULT_KERNEL_PARAMS.copy()
        self.kernel_non_normalize_params = (
            non_normalize_params or self.DEFAULT_NON_NORMALIZE_PARAMS.copy()
        )
        self.scaler = None

    @property
    def kernel_params(self):
        if self.scaler is None:
            return {**self.kernel_normalize_params, **self.kernel_non_normalize_params}
        else:
            normalized_params = {
                key: self._normalize_hyperparam(value)
                for key, value in self.kernel_normalize_params.items()
            }
            return {**normalized_params, **self.kernel_non_normalize_params}

    def set_scaler(self, X: np.ndarray, method: str = "standard"):
        """
        Set the scaler based on the fitting data.
        :param X: Fitting data (2D array).
        :param method: Normalization method, either 'standard' or 'minmax'.
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported normalization method: {}".format(method))

        self.scaler.fit(X)

    def _normalize_hyperparam(self, hyperparam):
        """
        Normalize a hyperparameter based on the scaler.
        """
        if self.scaler is None:
            return hyperparam

        if isinstance(hyperparam, tuple):
            return tuple(np.asarray(hyperparam) / self.scaler.scale_[0])
        elif isinstance(hyperparam, (float, int)):
            return hyperparam / self.scaler.scale_[0]
        return hyperparam

    def make_kernel(self, normalize: bool = True) -> Kernel:
        """
        Create a kernel using the specified function and parameters.
        :param normalize: Whether to normalize the parameters.
        :return: Kernel object.
        """
        if not normalize or not self.scaler:
            all_params = {**self.kernel_params, **self.kernel_non_normalize_params}
            return self.kernel_function(**all_params)

        return self.kernel_function(**self.kernel_params)

    def update_kernel_params(self, new_params: dict, normalize: bool = True) -> "KernelManager":
        """
        Update the kernel parameters.
        :param new_params: New parameters to be used.
        """
        if not normalize or not self.scaler:
            self.kernel_params.update(new_params)
            return self
        else:
            normalized_params = {
                key: self._normalize_hyperparam(value) for key, value in new_params.items()
            }
            self.kernel_params.update(normalized_params)
            return self
