from typing import Callable, Union, Optional
import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .custom_scikit_kernel import StepKernel


class KernelManager:
    DEFAULT_KERNEL_FUNCTION: Callable[[float, tuple, float], Kernel] = staticmethod(
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
        kernel_function: Optional[Callable[[float, tuple, float], Kernel]] = None,
        kernel_params: Optional[dict] = None,
        non_normalize_params: Optional[dict] = None,
        max_range: Optional[Union[float, int]] = None,
    ):
        if kernel_params is None:
            kernel_params = self.DEFAULT_KERNEL_PARAMS.copy()
        if non_normalize_params is None:
            non_normalize_params = self.DEFAULT_NON_NORMALIZE_PARAMS.copy()
        if kernel_function is None:
            kernel_function = self.DEFAULT_KERNEL_FUNCTION

        self.kernel_function = kernel_function
        self.kernel_normalize_params = kernel_params
        self.kernel_non_normalize_params = non_normalize_params

        self.max_range = max_range
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

    def set_scaler(self, scaler: Union[StandardScaler, MinMaxScaler]) -> "KernelManager":
        """
        Set the scaler to be used for normalization.

        :param scaler: Scaler object.
        :return: The updated KernelManager instance.
        """
        self.scaler = scaler
        return self

    def set_scaler_from_X(self, X: np.ndarray, method: str = "standard") -> None:
        """
        Set the scaler based on the fitting data.

        :param X: Fitting data (2D array).
        :param method: Normalization method, either 'standard' or 'minmax'.
        :raises ValueError: If an unsupported normalization method is provided.
        """
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
        }
        if method not in scalers:
            raise ValueError(f"Unsupported normalization method: {method}")

        self.scaler = scalers[method]
        self.scaler.fit(X)

    def _normalize_hyperparam(
        self, hyperparam: Union[float, int, tuple]
    ) -> Union[float, int, tuple]:
        """
        Normalize a hyperparameter based on the scaler.

        :param hyperparam: Hyperparameter value.
        :return: Normalized hyperparameter value.
        """
        if self.scaler is None:
            return hyperparam

        if isinstance(hyperparam, tuple):
            return tuple(np.asarray(hyperparam) / self.scaler.scale_[0])
        elif isinstance(hyperparam, (float, int)):
            return hyperparam / self.scaler.scale_[0]
        return hyperparam

    def make_kernel(
        self, normalize: bool = True, max_range: Optional[Union[float, int]] = None
    ) -> Kernel:
        """
        Create a kernel using the specified function and parameters.

        :param normalize: Whether to normalize the parameters.
        :param max_range: Maximum range for the kernel.
        :return: Kernel object.
        """
        if not normalize or not self.scaler:
            all_params = {**self.kernel_params, **self.kernel_non_normalize_params}
            kernel = self.kernel_function(**all_params)

        kernel = self.kernel_function(**self.kernel_params)

        if max_range is None:
            max_range = self.max_range

        if max_range:
            # normalize self.max_range
            if self.scaler:
                max_range = max_range / self.scaler.scale_[0]
            else:
                max_range = max_range
            kernel *= StepKernel(max_range)

        return kernel

    def update_kernel_params(self, new_params: dict, normalize: bool = True) -> "KernelManager":
        """
        Update the kernel parameters.

        :param new_params: New parameters to be used.
        :param normalize: Whether to normalize the parameters.
        :return: The updated KernelManager instance.
        """
        if not normalize or not self.scaler:
            self.kernel_normalize_params.update(new_params)
        else:
            normalized_params = {
                key: self._normalize_hyperparam(value) for key, value in new_params.items()
            }
            self.kernel_normalize_params.update(normalized_params)
        return self
