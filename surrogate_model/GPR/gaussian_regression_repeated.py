import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from .gaussian_regression import GaussianRegressionModel

class GaussianRegressionModelWithRepeats(GaussianRegressionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, y_std=None, **kwargs) -> "GaussianRegressionModelWithRepeats":
        """
        Fit the Gaussian Process model, considering the standard deviation of repeated measurements.

        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param y: Observed field values at these coordinates (shape [n_samples]).
        :param y_std: Standard deviations for the observations (shape [n_samples]).
                      If None, a uniform standard deviation is assumed.
        """
        if y_std is None:
            y_std = np.full_like(y, fill_value=self.alpha)

        # Scale the input data
        X = X / self.units.value
        X_scaled = self.scaler.fit_transform(X)

        # Update the kernel to include a WhiteKernel component for the noise
        # noise_kernel = WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 0.00001))
        self.kernel = self.kernelM.make_kernel(normalize=True) 

        
        
        # Fit the Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            alpha=1e-10,  # Small value to ensure numerical stability
            **kwargs,
        )
        self.gp.fit(X_scaled, y)
        return self

