from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Callable, Literal, Union
from logging import Logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from ...kriging.variogram_analysis import VariogramAnalyzer
from ...kriging.ordinary_kriging import OrdinaryKrigingEstimator

from ..surrogate_model import SurrogateModel
from ..abstract_sample_model import AbstractSampleModel

from ._ok_plotter_mixin import OKPlotterMixinClass

from my_packages.constants import DistanceUnits


@dataclass
class OrdinaryKrigingModelConfigs:
    units: DistanceUnits = DistanceUnits.mm
    max_range: float = 30
    variogram_angle: Optional[int] = None
    angle_tolerance: float = 10
    lag_step: int = 3
    variogram_model_type: Optional[
        Literal["linear", "power", "gaussian", "exponential", "spherical", "hole_effect"]
    ] = None
    n_closest_points: Optional[int] = None
    backend: Literal["vectorized", "loop"] = "vectorized"
    normalize: bool = True

    def __post_init__(self):
        if isinstance(self.units, str):
            self.units = DistanceUnits[self.units]
        if isinstance(self.variogram_model_type, str):
            if self.variogram_model_type not in [
                "linear",
                "power",
                "gaussian",
                "exponential",
                "spherical",
                "hole_effect",
            ]:
                raise ValueError(
                    "Invalid variogram model type. Choose from 'linear', 'power', 'gaussian', 'exponential', 'spherical', 'hole_effect'"
                )
        # if self.variogram_angle is None:
        #     # if angle is not provided, set it to 0
        #     # and set the tolerance to 90 degrees
        #     # i.e. the variogram will be isotropic
        #     self.variogram_angle = 0
        #     self.angle_tolerance = 90


class OrdinaryKrigingModel(AbstractSampleModel, OKPlotterMixinClass):
    def __init__(self, configs: OrdinaryKrigingModelConfigs, **kwargs):
        """
        Initialize the OrdinaryKrigingModel.

        Args:
            configs (OrdinaryKrigingModelConfigs): Configuration dataclass for the model.
            **kwargs: Additional keyword arguments to update the configs.
        """
        self.logger = Logger(__name__)
        # update the configs
        self.configs = configs
        for key, value in kwargs.items():
            if hasattr(self.configs, key):
                setattr(self.configs, key, value)
            else:
                self.logger.warning(f"Invalid config key: {key}")
        if self.configs.normalize:
            self._set_scalers()
        self.variogram_analyzer_ = None
        self.kriging_estimator = None

    def _set_scalers(self):
        """
        Set the label scaler and position scaler for normalizing the target values and positions.
        """
        self.label_scaler = StandardScaler()
        self.position_scaler = StandardScaler(with_std=False)

    @property
    def scale_factor(self):
        if self.configs.normalize:
            if self.position_scaler.scale_ is not None:
                return np.mean(self.position_scaler.scale_)
        return 1    

    @property
    def lags(self):
        if not hasattr(self, "lags_"):
            return None
        return self.lags_ * self.configs.units.value

    @property
    def semivariances(self):
        if not hasattr(self, "semivariances_"):
            return None
        return self.semivariances_ * self.label_scaler.scale_**2

    @property
    def variogram_analyzer(self):       
        analyzer = VariogramAnalyzer(
            positions=self.X_train,
            values=self.y_train.flatten(),
        )      
        if self.variogram_analyzer_.lags is not None:
            analyzer.lags = self.lags
        if self.variogram_analyzer_.deltas_ is not None:
            rescaled_deltas = []
            for delta_tuple in self.variogram_analyzer_.deltas_:
                rescaled_deltas.append(
                    (
                        delta_tuple[0] * self.configs.units.value, 
                        delta_tuple[1] * self.configs.units.value
                    )
                )
            analyzer.deltas_ = rescaled_deltas
        if self.variogram_analyzer_.semivariances is not None:
            analyzer.semivariances = self.semivariances
        if self.variogram_analyzer_.angles_ is not None:
            analyzer.angles_ = self.variogram_analyzer_.angles_
        if self.variogram_analyzer_.tolerances_ is not None:
            analyzer.tolerances_ = self.variogram_analyzer_.tolerances_
        return analyzer

    def fit_variogram(
        self, X: np.ndarray, y: np.ndarray
    ):
        """
        Fit the variogram model to the given data.

        Args:
            X (np.ndarray): Input positions.
            y (np.ndarray): Target values.
            initial_params (Optional[Dict[str, float]]): Initial parameters for the variogram model.
        """

        self.X_train = X
        self.y_train = y

        if self.configs.normalize:
            self.X_scaled = self.position_scaler.fit_transform(self.X_train)
            self.y_scaled = self.label_scaler.fit_transform(y.reshape(-1, 1))
        else:
            self.X_scaled = self.X_train
            self.y_scaled = self.y_train

        # transform to position values in units
        self.X_scaled = self.X_scaled / self.configs.units.value

        self.variogram_analyzer_ = VariogramAnalyzer(
            positions=self.X_scaled,
            values=self.y_scaled.flatten(),
        )

        # # variogram properties
        # lag_step = self.configs.lag_step * self.configs.units.value
        # max_range = self.configs.max_range * self.configs.units.value

        # # normalize the lag_step and max_range
        # normalized_lag_step = lag_step / self.scale_factor
        # normalized_max_range = max_range / self.scale_factor

        self.lags_, self.semivariances_ = self.variogram_analyzer_.calculate_empirical_variogram(
            lag_step=self.configs.lag_step,
            max_range=self.configs.max_range,
            angle=self.configs.variogram_angle,
            tolerance=self.configs.angle_tolerance,
        )

        self.variogram_model = self.variogram_analyzer_.fit_variogram(
            model_type=self.configs.variogram_model_type
        )
        return self

    def update_variogram(self, variogram_model: Dict[str, float]):
        """
        Update the variogram model with new parameters.

        Args:
            variogram_model (Dict[str, float]): Updated variogram model parameters.
        """
        self.variogram_model = variogram_model

    def predict(self, X_predict: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Predict the target values for new positions using Ordinary Kriging.

        Args:
            X_new (np.ndarray): New positions for prediction.
            return_std (bool): Whether to return the standard deviation of the predictions.

        Returns:
            np.ndarray: Predicted target values (and standard deviations if return_std is True).
        """

        if np.ndim(X_predict) > 2:
            if X_predict.shape[0] == 3: 
                X_predict = X_predict[:2]
            if X_predict.shape[0] != 2:
                raise ValueError("X_predict must be a 2D array with shape (n_samples, 2).")
            X_predict = X_predict.reshape(2, -1).T

        if self.configs.normalize:
            X_new_scaled = self.position_scaler.transform(X_predict)
        else:
            X_new_scaled = X_predict

        X_new_scaled = X_new_scaled / self.configs.units.value

        self.kriging_estimator = OrdinaryKrigingEstimator(
            variogram_model=self.variogram_model,
            data=self.y_scaled.squeeze(),
            sample_points=self.X_scaled,
        )
        y_predicted, var_predicted = self.kriging_estimator.estimate(
            estimation_points=X_new_scaled,
            n_closest_points=self.configs.n_closest_points,
            backend=self.configs.backend,
        )
        if self.configs.normalize:
            y = self.label_scaler.inverse_transform(y_predicted.reshape(-1, 1))
        else:
            y = y_predicted
        if return_std:
            if self.configs.normalize:
                var = var_predicted * (self.label_scaler.scale_**2)
            else:
                var = var_predicted 
            std = np.sqrt(var)
            return y, std
        return y

    @property
    def surrogate_model(self):
        return SurrogateModel(self)
