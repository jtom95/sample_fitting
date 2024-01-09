from dataclasses import dataclass
from typing import Tuple, List
from collections import namedtuple
import numpy as np
import numpy.polynomial.chebyshev as cheb

from .abstract_curve_fitter import AbstractCurveFitter

Normalization_Info = namedtuple("Normalization_Info", ["x_min", "x_max", "y_min", "y_max"])


@dataclass
class Normalizer:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    # normalized: bool = False

    def normalize(self, x: np.ndarray, y: np.ndarray):
        """
        Normalizes the inputs by scaling them between -1 and 1 (if self.x_min == min(x) etc.).
        """
        x_norm = self.normalize_x(x)
        y_norm = self.normalize_y(y)
        return x_norm, y_norm

    def normalize_x(self, x: np.ndarray):
        """
        Normalizes the inputs by scaling them between -1 and 1 (if self.x_min == min(x) etc.).
        """

        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        return x_norm

    def normalize_y(self, y: np.ndarray):
        """
        Normalizes the inputs by scaling them between -1 and 1 (if self.x_min == min(x) etc.).
        """

        y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        return y_norm

    def denormalize_y(self, y):
        return self.y_min + (self.y_max - self.y_min) * (y + 1) / 2

    def denormalize_x(self, x):
        return self.x_min + (self.x_max - self.x_min) * (x + 1) / 2

    @classmethod
    def from_normalization_info(cls, info: Normalization_Info):
        return cls(**info._asdict())

    def get_normalization_info(self):
        return Normalization_Info(
            x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max
        )


class ChebyshevFitter1D(AbstractCurveFitter):
    """
    Class to perform vector fitting using Chebyshev basis.

    This class implements methods for loading data, normalizing data, fitting a Chebyshev
    basis function to the data, and predicting values for new data points.

    Attributes:
        x_min (float): The minimum bound of the spatial domain.
        x_max (float): The maximum bound of the spatial domain.
        x (np.ndarray): The positions of the data points.
        y (np.ndarray): The values of the data points.
        coefficients (np.ndarray): The Chebyshev coefficients.
        normalizer (Normalizer): The normalizer used to normalize the data.
    """

    def __init__(self, bounds: Tuple[float, float] = None):
        """
        Initializes the class with an optional bounds argument.

        Args:
            bounds (Tuple[float, float]): The minimum and maximum bounds of the spatial domain.
        """

        if bounds is not None:
            self._x_min = bounds[0]
            self._x_max = bounds[1]
        else:
            self._x_min = None
            self._x_max = None

        self.x = None
        self.y = None
        self.coefficients = None
        self.normalizer = None
    
    @property
    def x_min(self):
        if self._x_min is None:
            if self.x is None:
                return None
            else:
                return np.min(self.x)
        else:
            return self._x_min
    
    @property
    def x_max(self):
        if self._x_max is None:
            if self.x is None:
                return None
            else:
                return np.max(self.x)
        else:
            return self._x_max

    def load_data(self, positions: np.ndarray, values: np.ndarray):
        """
        Loads the data to be fitted.

        Args:
            positions (np.ndarray): The positions of the data points.
            values (np.ndarray): The values of the data points.
        """
        self.check_data_1d(positions, values)
        self.x = positions
        self.y = values

    def normalize(self):
        """
        Normalizes the input function by scaling it to the interval [-1, 1].

        This method normalizes both the `x` and `y` data using a `Normalizer` object.
        Normalization is done to improve numerical stability and to ensure that the Chebyshev
        basis functions are properly scaled.

        This method also sets the `normalizer` attribute.

        Returns:
            ChebyshevFitter1D: The ChebyshevFitter1D object with the normalized data.
        """


        y_min = np.min(self.y)
        y_max = np.max(self.y)        

        norm_info = Normalization_Info(x_min=self.x_min, x_max=self.x_max, y_min=y_min, y_max=y_max)

        self.normalizer = Normalizer.from_normalization_info(norm_info)

        # normalize the input data
        self.x_norm, self.y_norm = self.normalizer.normalize(self.x, self.y)

        return self.x_norm, self.y_norm

    @staticmethod
    def chebyshev_basis(x: np.ndarray, order: int = 10):
        """
        Computes the Chebyshev basis functions of order n.

        The Chebyshev basis functions are a set of orthogonal polynomials
        that can be used to represent a function on the interval [-1, 1].
        This method computes the first n (= to `order`) Chebyshev basis functions evaluated at
        the given x values.

        Args:
            x (np.ndarray): The x values at which to evaluate the Chebyshev basis functions.
            order (int): The order of the Chebyshev basis functions.

        Returns:
            np.ndarray: A numpy array containing the Chebyshev basis functions evaluated at the given x values.
        """

        return cheb.chebvander(x, order)

    def fit(self, order: int = 10):
        """
        Performs vector fitting using Chebyshev basis functions of order n.

        This method fits the Chebyshev basis functions to the data and computes the coefficients.
        It also sets the `coefficients` and `normalizer` attributes.

        Args:
            order (int): The order of the Chebyshev basis functions.

        Returns:
            Tuple[np.ndarray, normalization_info]: A tuple containing the Chebyshev coefficients
            and the normalization information.
        """
        x_norm, y_norm = self.normalize()

        basis_matrix = self.chebyshev_basis(x_norm, order)

        # Use least squares regression to compute the coefficients
        self.coefficients, _, _, _ = np.linalg.lstsq(basis_matrix, y_norm, rcond=None)

        # Return the coefficients and the normalization information
        return (self.coefficients, self.normalizer.get_normalization_info())
    
    def predict(self, x):
        if not hasattr(self, "coefficients"):
            raise ValueError("The model is not fitted yet. Please call fit() first.")
        if not hasattr(self, "normalizer"):
            raise ValueError("The model is not fitted yet. Please call fit() first.")
        coeffs = self.coefficients
        norm_info = self.normalizer.get_normalization_info()
        
        return self.cls_predict(x, coeffs, norm_info)

    @classmethod
    def cls_predict(cls, positions, coefficients: np.ndarray, normalization_info: Normalization_Info):
        """
        Predicts the y value for a given x value using the fitted Chebyshev basis functions.

        Args:
            x: The x value for which to predict the y value.

        Returns:
            The predicted y value.
        """

        normalizer = Normalizer.from_normalization_info(normalization_info)

        # Normalize positions
        positions_normalized = normalizer.normalize_x(positions)

        # Compute the Chebyshev basis functions
        basis_matrix = cheb.chebvander(positions_normalized, len(coefficients) - 1)

        # Calculate the y values as the dot product of the basis matrix and coefficients
        y_normalized = basis_matrix @ coefficients

        # Denormalize y values
        y_pred = normalizer.denormalize_y(y_normalized)

        return y_pred
