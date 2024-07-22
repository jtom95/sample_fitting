from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Callable, Literal, Union
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist


from ...kriging.variogram_analysis import VariogramAnalyzer
from ...kriging.ordinary_kriging import OrdinaryKrigingEstimator

from ..surrogate_model import SurrogateModel
from ..abstract_sample_model import AbstractSampleModel

from ._ok_plotter_mixin import OKPlotterMixinClass

from my_packages.EM_fields.scans import Scan, Grid
from my_packages.constants import DistanceUnits


@dataclass
class OrdinaryKrigingModelConfigs:
    units: DistanceUnits = DistanceUnits.mm
    max_range: float = 30
    variogram_angle: Optional[int] = None
    angle_tolerance: float = 10
    lag_step: int = 3
    variogram_model_type: Optional[
        Literal["gaussian", "exponential", "spherical", "hole_effect"]
    ] = None
    n_closest_points: Optional[int] = None
    backend: Literal["vectorized", "loop"] = "vectorized"
    normalize: bool = True

    def __post_init__(self):
        if isinstance(self.units, str):
            self.units = DistanceUnits[self.units]
        if isinstance(self.variogram_model_type, str):
            if self.variogram_model_type not in [
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
        self.logger = logging.getLogger(__name__)
        # update the configs
        self.configs = configs
        self.reset(**kwargs)
        if self.configs.normalize:
            self._set_scalers()
        
    
    def reset(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.configs, key):
                setattr(self.configs, key, value)
            else:
                self.logger.warning(f"Invalid config key: {key}")
        self.variogram_analyzer_ = None
        self.kriging_estimator = None

        self.estimation_points_ = None
        self.estimated_values_ = None
        self.estimation_variances_ = None
        return self

    def _set_scalers(self):
        """
        Set the label scaler and position scaler for normalizing the target values and positions.
        """
        self.label_scaler = StandardScaler()
        self.position_scaler = StandardScaler(with_std=False)

    def initialize_kriging(self):
        self.kriging_estimator = OrdinaryKrigingEstimator(
            variogram_model=self.variogram_model,
            data=self.y_scaled.squeeze(),
            sample_points=self.X_scaled,
        )

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
    def nugget(self):
        if not hasattr(self, "variogram_model"):
            return None
        if self.variogram_model is None:
            return None
        if self.variogram_model.get("nugget") is None:
            return None
        return self.variogram_model.get("nugget") * self.label_scaler.scale_**2

    @property
    def sill(self):
        if not hasattr(self, "variogram_model"):
            return None
        if self.variogram_model is None:
            return None
        if self.variogram_model.get("sill") is None:
            return None
        return self.variogram_model.get("sill") * self.label_scaler.scale_**2

    @property
    def range_(self):
        if not hasattr(self, "variogram_model"):
            return None
        if self.variogram_model is None:
            return None
        if self.variogram_model.get("range_") is None:
            return None
        return self.variogram_model.get("range_") * self.configs.units.value * self.scale_factor

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
                        delta_tuple[0] * self.configs.units.value * self.scale_factor, 
                        delta_tuple[1] * self.configs.units.value * self.scale_factor
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
        self, X: np.ndarray, y: np.ndarray, range_: Optional[float] = None, sill: Optional[float] = None, nugget: Optional[float] = None
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

            if range_ is not None:
                range_ = range_ / self.scale_factor
            if sill is not None:
                sill = sill / self.label_scaler.scale_**2
            if nugget is not None:
                nugget = nugget / self.label_scaler.scale_**2
        else:
            self.X_scaled = self.X_train
            self.y_scaled = self.y_train

        # transform to position values in units
        self.X_scaled = self.X_scaled / self.configs.units.value
        range_ = range_ / self.configs.units.value if range_ is not None else None

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
            model_type=self.configs.variogram_model_type, nugget=nugget, sill=sill, range_=range_
        )

        # initialize the kriging estimator
        self.initialize_kriging()
        return self

    def update_variogram(self, variogram_model: Dict[str, float]):
        """
        Update the variogram model with new parameters.

        Args:
            variogram_model (Dict[str, float]): Updated variogram model parameters.
        """
        self.variogram_model = variogram_model
        self.initialize_kriging()

    def estimate(
        self, estimation_points: np.ndarray, force_reestimate: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the target values and variances for the given estimation points using Ordinary Kriging.

        Args:
            estimation_points (np.ndarray): Estimation points for kriging.
            force_reestimate (bool): Whether to force re-estimation even if previous results are available.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Estimated target values and variances.
        """
        if np.ndim(estimation_points) > 2:
            if estimation_points.shape[0] == 3:
                estimation_points = estimation_points[:2]
            if estimation_points.shape[0] != 2:
                raise ValueError("estimation_points must be a 2D array with shape (n_samples, 2).")
            estimation_points = estimation_points.reshape(2, -1).T

        if self.configs.normalize:
            estimation_points_scaled = self.position_scaler.transform(estimation_points)
        else:
            estimation_points_scaled = estimation_points

        estimation_points_scaled = estimation_points_scaled / self.configs.units.value

        if (
            force_reestimate
            or self.estimation_points_ is None
            or not np.array_equal(self.estimation_points_, estimation_points_scaled)
        ):
            self.estimation_points_ = estimation_points_scaled
            self.estimated_values_, self.estimation_variances_ = self.kriging_estimator.estimate(
                estimation_points=self.estimation_points_,
                n_closest_points=self.configs.n_closest_points,
                backend=self.configs.backend,
            )

        if self.configs.normalize:
            estimated_values = self.label_scaler.inverse_transform(
                self.estimated_values_.reshape(-1, 1)
            )
            estimation_variances = self.estimation_variances_ * (self.label_scaler.scale_**2)
        else:
            estimated_values = self.estimated_values_
            estimation_variances = self.estimation_variances_

        return estimated_values, estimation_variances

    def predict(self, X_predict: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Wrapper of the estimate method: needed for the SurrogateModel interface.
        
        Predict the target values for new positions using Ordinary Kriging.

        Args:
            X_new (np.ndarray): New positions for prediction.
            return_std (bool): Whether to return the standard deviation of the predictions.

        Returns:
            np.ndarray: Predicted target values (and standard deviations if return_std is True).
        """
        y_predicted, var_predicted = self.estimate(X_predict)

        if return_std:
            std = np.sqrt(var_predicted)
            return y_predicted, std
        return y_predicted

    def sample_y(self, X, n_samples=1, random_state=0):
        """
        Draw samples from the kriging model and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the kriging model is evaluated.
        n_samples : int, default=1
            Number of samples drawn from the kriging model per query point.
        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples)
            Values of n_samples samples drawn from the kriging model and evaluated at query points.
        """
        rng = np.random.RandomState(random_state)
        y_mean, y_var = self.estimate(X)
        y_std = np.sqrt(y_var)

        y_samples = rng.normal(loc=y_mean, scale=y_std, size=(X.shape[0], n_samples))
        return y_samples

    def sequential_simulate(
        self, simulation_points: np.ndarray, n_realizations: int = 1, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Perform sequential Gaussian simulation based on the estimated kriging model.

        Args:
            simulation_points (np.ndarray): The points at which to simulate values.
            n_realizations (int): The number of realizations to generate. Default is 1.
            seed (int, optional): The random seed for reproducibility. Default is None.

        Returns:
            np.ndarray: The simulated values at the simulation points. The shape will be (n_realizations, len(simulation_points)).
        """
        np.random.seed(seed)
        num_simulation_points = simulation_points.shape[0]
        simulated_values = np.zeros((n_realizations, num_simulation_points))

        # if self.configs.normalize:
        #     simulation_points_scaled = self.position_scaler.transform(simulation_points)
        # else:
        #     simulation_points_scaled = simulation_points

        # simulation_points_scaled = simulation_points_scaled / self.configs.units.value

        for realization in range(n_realizations):
            estimation_points = self.X_scaled.copy()
            estimation_values = self.y_scaled.copy().squeeze()

            # Generate a random permutation of indices for the simulation points
            simulation_indices = np.random.permutation(num_simulation_points)

            for i in simulation_indices:
                simulation_point = simulation_points[i]
                simulated_value = self.sample_y(simulation_point.reshape(1, -1), random_state=None)

                simulated_values[realization, i] = simulated_value

                estimation_points = np.vstack((estimation_points, simulation_point))
                estimation_values = np.append(estimation_values, simulated_value)

        if self.configs.normalize:
            simulated_values = self.label_scaler.inverse_transform(
                simulated_values.reshape(-1, 1)
            ).reshape(n_realizations, -1)

        return simulated_values

    def sequential_simulate_on_grid(self, grid: np.ndarray, n_realizations: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Perform sequential Gaussian simulation on a grid based on the estimated kriging model.

        Args:
            grid (np.ndarray): The grid points at which to simulate values.
                The grid should have shape (2, M, N), where the first dimension represents the x and y coordinates,
                and M and N are the dimensions of the grid.
            n_realizations (int): The number of realizations to generate. Default is 1.
            seed (int, optional): The random seed for reproducibility. Default is None.

        Returns:
            np.ndarray: The simulated values at the grid points. The shape will be (n_realizations, M, N).
        """
        if np.squeeze(grid).ndim == 3 and np.ndim(grid) == 4:
            grid = grid.squeeze()
        if grid.shape[0] == 3:
            grid = grid[:2]
        if grid.ndim != 3 or grid.shape[0] != 2:
            raise ValueError("The grid should have shape (2, M, N)")

        M, N = grid.shape[1], grid.shape[2]
        simulation_points = grid.reshape(2, -1).T

        simulated_values = self.sequential_simulate(
            simulation_points=simulation_points,
            n_realizations=n_realizations,
            seed=seed
        )

        simulated_values = simulated_values.reshape(n_realizations, M, N)

        return simulated_values

    @property
    def surrogate_model(self):
        return SurrogateModel(self)
