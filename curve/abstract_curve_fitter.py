from abc import ABC, abstractmethod, abstractproperty
import numpy as np


class AbstractCurveFitter(ABC):
    """
    Abstract base class for curve fitting.

    This class defines the interface for curve fitting classes. It requires
    the implementation of fit, predict, and a method to check if the data is 1D.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits a curve to the given data.

        Args:
            x (np.ndarray): The x values of the data points.
            y (np.ndarray): The y values of the data points.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts y values for the given x values using the fitted curve.

        Args:
            x (np.ndarray): The x values for which to predict y values.

        Returns:
            np.ndarray: The predicted y values.
        """
        pass

    def check_data_1d(self, x: np.ndarray, y: np.ndarray):
        """
        Checks if the given data arrays are 1-dimensional.

        Args:
            x (np.ndarray): The x values of the data points.
            y (np.ndarray): The y values of the data points.

        Raises:
            ValueError: If x or y is not 1-dimensional.
        """
        if not (x.ndim == 1 and y.ndim == 1):
            raise ValueError("Both x and y must be 1-dimensional arrays.")
    
    @abstractproperty
    def x_min(self):
        pass
    @abstractproperty
    def x_max(self):
        pass