from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
    
from .abstract_curve_fitter import AbstractCurveFitter

class UnivariateFitting(AbstractCurveFitter):
    def __init__(self, spline_order=3, smoothing_factor=None):
        """
        Initialize the UnivariateFitting with a specified spline order and smoothing factor.

        Args:
            spline_order (int): The order of the spline (default is cubic, 3).
            smoothing_factor (float): The smoothing factor for the spline.
        """
        self.spline_order = spline_order
        self.smoothing_factor = smoothing_factor
        self.spline = None
        self.x_tr = None
        self.y = None
        self.x_scaler = StandardScaler()
        self.y_scaler = MinMaxScaler()
        
    
    @property
    def x_min(self):
        if self.x_tr is None:
            return None
        return self.x_tr.min()

    @property
    def x_max(self):
        if self.x_tr is None:
            return None
        return self.x_tr.max()


    def fit(self, x: np.ndarray, y: np.ndarray, 
            spline_order: float=None, smoothing_factor: float=None, 
            **kwargs
            ):
        """
        Fits a univariate spline to the given data.
        Overrides the abstract method in AbstractCurveFitter.
        """
        self.check_data_1d(x, y)
        
        if spline_order is None:
            spline_order = self.spline_order
        
        if smoothing_factor is None:
            smoothing_factor = self.smoothing_factor
        
        self.x_tr = x
        self.y = y

        # Normalize the data
        x_normalized = self.x_scaler.fit_transform(x.reshape(-1, 1)).flatten()
        y_normalized = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        def_kwargs = {
            "ext": 2, # Extrapolation raises an error
        }
        
        kwargs = {**def_kwargs, **kwargs}

        # Fit the spline
        self.spline = UnivariateSpline(
            x_normalized, y_normalized, 
            k=spline_order, s=smoothing_factor,
            **kwargs
            )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts y values for the given x values using the fitted spline.
        Overrides the abstract method in AbstractCurveFitter.
        """
        if self.spline is None:
            raise ValueError("The spline model is not fitted yet. Please call fit() first.")

        # Normalize x values
        x_normalized = self.x_scaler.transform(x.reshape(-1, 1)).flatten()

        # Predict and denormalize y values
        y_normalized = self.spline(x_normalized)
        y_pred = self.y_scaler.inverse_transform(y_normalized.reshape(-1, 1)).flatten()

        return y_pred