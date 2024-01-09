from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

from .abstract_curve_fitter import AbstractCurveFitter


class InterpolationFitting(AbstractCurveFitter):
    def __init__(self, kind="cubic", fill_value="extrapolate", x_min=None, x_max=None):
        """
        Initialize the InterpolationFitting with a specified kind of interpolation.

        Args:
            kind (str or int): Specifies the kind of interpolation as a string
                ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.)
                or as an integer specifying the order of the spline interpolator.
            fill_value: Value to use for points outside of the interpolation domain.
        """
        self.kind = kind
        self.fill_value = fill_value
        self.interpolator = None
        self.x_scaler = StandardScaler()
        self.y_scaler = MinMaxScaler()
        
        self.x = None
        self._x_min = x_min
        self._x_max = x_max
    
    @property
    def x_min(self):
        if self._x_min is None:
            if self.x is None:
                return None
            else:
                return np.min(self.x)
    
    @property
    def x_max(self):
        if self._x_max is None:
            if self.x is None:
                return None
            else:
                return np.max(self.x)
        else:
            return self._x_max
            

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fits an interpolation model to the given data.
        Overrides the abstract method in AbstractCurveFitter.
        """
        self.check_data_1d(x, y)
        
        # save the training data
        self.x = x

        # Normalize the data
        x_normalized = self.x_scaler.fit_transform(x.reshape(-1, 1)).flatten()
        y_normalized = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Fit the interpolator
        self.interpolator = interp1d(
            x_normalized, y_normalized, kind=self.kind, fill_value=self.fill_value,
            **kwargs
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts y values for the given x values using the fitted interpolation model.
        Overrides the abstract method in AbstractCurveFitter.
        """
        if self.interpolator is None:
            raise ValueError("The interpolation model is not fitted yet. Please call fit() first.")

        # Normalize x values
        x_normalized = self.x_scaler.transform(x.reshape(-1, 1)).flatten()

        # Predict and denormalize y values
        y_normalized = self.interpolator(x_normalized)
        y_pred = self.y_scaler.inverse_transform(y_normalized.reshape(-1, 1)).flatten()

        return y_pred
