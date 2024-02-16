import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, griddata
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from ..abstract_sample_model import AbstractSampleModel
from ..surrogate_model import SurrogateModel


class Interpolator(AbstractSampleModel):
    def __init__(self, method="cubic", normalization_method="standard"):
        """
        Initializes the SplineGeneralModel.
        :param spline_order: Order of the spline (e.g., 3 for cubic splines).
        """
        self.interpolation_method = method
        self.scaler = StandardScaler() if normalization_method == "standard" else MinMaxScaler()
        self.X_fitted_scaled = None
        self.z_real = None
        self.z_imag = None
        self.z = None
        
    @property
    def surrogate_model(self) -> SurrogateModel:
        return SurrogateModel(self)

    def fit(self, X, y):
        """
        Fit the spline model to the data.
        :param X: 2D array of shape (n_samples, 2) for x and y coordinates.
        :param y: 1D array of n_samples.
        """
        if X.shape[1] != 2:
            raise ValueError("X must be a 2D array with two columns for x and y coordinates.")

        # Normalizing the data
        self.X_fitted_scaled = self.scaler.fit_transform(X)
        z = y

        # check if the data is complex
        if np.any(np.iscomplex(y)):
            self.z_real = np.real(z)
            self.z_imag = np.imag(z)
        else:
            self.z = z
        return self
    
    @staticmethod
    def fill_nan(array_1d: np.ndarray) -> np.ndarray:
        series = pd.Series(array_1d)
        # fill the nan values with the closest non-nan value
        series = series.bfill()
        return series.values
    
    def predict(self, X):
        """
        Make predictions using the spline model.
        :param X: 2D array of shape (n_samples, 2) for x and y coordinates.
        :return: Predicted complex values.
        """
        if self.X_fitted_scaled is None:
            raise ValueError("Model must be fitted before prediction.")

        # Normalizing the prediction data
        X_scaled = self.scaler.transform(X)
        x_scaled = X_scaled[:, 0]
        y_scaled = X_scaled[:, 1]

        if self.z_real is not None and self.z_imag is not None:
            z_real = griddata(self.X_fitted_scaled, self.z_real, (x_scaled, y_scaled), method=self.interpolation_method)
            z_imag = griddata(self.X_fitted_scaled, self.z_imag, (x_scaled, y_scaled), method=self.interpolation_method)
            
            # fill the nan values with the closest non-nan value
            z_real = self.fill_nan(z_real)
            z_imag = self.fill_nan(z_imag)
            
            return z_real + 1j * z_imag
        else:
            interp = griddata(self.X_fitted_scaled, self.z, (x_scaled, y_scaled), method=self.interpolation_method)
            # fill the nan values with the closest non-nan value
            interp = self.fill_nan(interp)
            return interp

class SplineOnRegularGridModel(AbstractSampleModel):
    def __init__(self, spline_order=3, normalization_method="standard"):
        """
        Initializes the SplineOnRegularGridModel.
        :param spline_order: Order of the spline (e.g., 3 for cubic splines).
        """
        self.spline_order = spline_order
        self.spline_real = None
        self.spline_imag = None
        self.spline = None
        self.scaler = StandardScaler() if normalization_method == "standard" else MinMaxScaler()

    @property
    def surrogate_model(self) -> SurrogateModel:
        return SurrogateModel(self)

    def fit(self, X, y) -> "SplineOnRegularGridModel":
        """
        Fit the spline model to the data.
        :param X: 2D array of shape (n_samples, 2) for x and y coordinates.
        :param y: 1D array of n_samples.
        """
        if X.shape[1] != 2:
            raise ValueError("X must be a 2D array with two columns for x and y coordinates.")

        # Normalizing the data
        X_scaled = self.scaler.fit_transform(X)
        x_scaled = X_scaled[:, 0]
        y_scaled = X_scaled[:, 1]
        z = y

        # Creating a meshgrid for the interpolation
        x_unique = np.unique(x_scaled)
        y_unique = np.unique(y_scaled)

        # check if the data is complex
        if np.any(np.iscomplex(y)):
            z_real = np.real(z)
            z_imag = np.imag(z)

            self.spline_real = RectBivariateSpline(
                x_unique,
                y_unique,
                z_real.reshape((len(x_unique), len(y_unique))),
                kx=self.spline_order,
                ky=self.spline_order,
            )
            self.spline_imag = RectBivariateSpline(
                x_unique,
                y_unique,
                z_imag.reshape((len(x_unique), len(y_unique))),
                kx=self.spline_order,
                ky=self.spline_order,
            )
        else:
            self.spline = RectBivariateSpline(
                x_unique,
                y_unique,
                z.reshape((len(x_unique), len(y_unique))),
                kx=self.spline_order,
                ky=self.spline_order,
            )
            self.spline_imag = None
        return self

    def predict(self, X):
        """
        Make predictions using the spline model.
        :param X: 2D array of shape (n_samples, 2) for x and y coordinates.
        :return: Predicted complex values.
        """
        if self.spline is None and (self.spline_real is None or self.spline_imag is None):
            raise ValueError("Model must be fitted before prediction.")

        # Normalizing the prediction data
        X_scaled = self.scaler.transform(X)
        x_scaled = X_scaled[:, 0]
        y_scaled = X_scaled[:, 1]

        if self.spline_real is not None and self.spline_imag is not None:
            z_real = self.spline_real.ev(x_scaled, y_scaled)
            z_imag = self.spline_imag.ev(x_scaled, y_scaled)
            return z_real + 1j * z_imag
        else:
            return self.spline.ev(x_scaled, y_scaled)

    def prediction_std(self, X):
        """
        Return a zero array for standard deviation as splines do not natively support uncertainty estimation.
        :param X: 2D array for x and y coordinates.
        :return: Array of zeros.
        """
        return np.zeros(len(X))
