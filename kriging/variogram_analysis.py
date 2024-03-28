from typing import Tuple, Optional, List, Dict, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


from .variogram_models import VariogramModels
from ._variogram_analysis_plotter_mixin import VariogramAnalyzerPlottingMixin
# from my_packages.auxiliary_plotting_functions.composite_plots import stack_figures
from my_packages.classes.aux_classes import Grid


class VariogramAnalyzer(VariogramAnalyzerPlottingMixin):
    """
    A class for analyzing variograms in 3D data.

    Attributes:
        data (numpy.ndarray): The 3D data array with dimensions x, y, f.
        position_grid (Grid, optional): The position grid corresponding to the data.
        frequency_indices (List[int]): The list of frequency indices to analyze.
    """

    def __init__(
        self,
        data: np.ndarray,
        position_grid: Optional[Grid] = None,
        frequency_indices: Optional[List[int]] = None,
    ):
        """
        Initialize the VariogramAnalyzer.

        Args:
            data (numpy.ndarray): The 3D data array with dimensions x, y, f.
            position_grid (Grid, optional): The position grid corresponding to the data. Defaults to None.
            frequency_indices (List[int], optional): The list of frequency indices to analyze. Defaults to None.
        """
        if np.ndim(data) == 2:
            self.data = data[:, :, np.newaxis]
        elif np.ndim(data) == 3:
            self.data = data
        else:
            raise ValueError("Data must be 2D or 3D.")
        self.position_grid = position_grid
        self.frequency_indices = (
            frequency_indices if frequency_indices is not None else list(range(data.shape[2]))
        )
        self.frequency_indices_order_dict = {
            freq_index: ii for ii, freq_index in enumerate(self.frequency_indices)
        }

        self.lags = None
        self.semivariances = None

    def fit_variogram_model_at_index(self, idx: int, model_type: str, initial_params: Optional[Tuple] = None) -> Dict | Callable:
        """
        Fit a variogram model to the empirical variogram data.

        Args:
            model_type (str): The type of variogram model to fit. Options: 'linear', 'power', 'gaussian', 'exponential', 'spherical', 'hole_effect'.
            initial_params (Tuple, optional): The initial parameters for the variogram model. Defaults to None.

        Returns:
            Dict: The fitted variogram model parameters.
        """
        if self.lags is None or self.semivariances is None:
            raise ValueError(
                "Empirical variogram data not available. Call calculate_empirical_variogram() first."
            )

        model_functions = {
            "linear": getattr(VariogramModels, "linear"),
            "power": getattr(VariogramModels, "power"),
            "gaussian": getattr(VariogramModels, "gaussian"),
            "exponential": getattr(VariogramModels, "exponential"),
            "spherical": getattr(VariogramModels, "spherical"),
            "hole_effect": getattr(VariogramModels, "hole_effect"),
        }

        if model_type not in model_functions:
            raise ValueError(
                f"Invalid model_type. Available options: {', '.join(model_functions.keys())}"
            )

        model_function = model_functions[model_type]

        if initial_params is None:
            if model_type == "linear":
                initial_params = (1, 0)  # Initial slope and nugget
            elif model_type == "power":
                initial_params = (1, 1, 0)  # Initial scale, exponent, and nugget
            else:
                initial_params = (
                    0,
                    np.max(self.semivariances),
                    np.max(self.lags),
                )  # Initial nugget, sill, and range

        position = self.frequency_indices_order_dict[idx]
        lags = self.lags[position] # chose the lags and semivariances for the specific frequency index
        semivariances = self.semivariances[position] # chose the lags and semivariances for the specific frequency index

        fitted_params, _ = curve_fit(
            model_function, lags.flatten(), semivariances.flatten(), p0=initial_params
        )

        if model_type == "linear":
            return_dict = {"slope": fitted_params[0], "nugget": fitted_params[1], "model_type": model_type}
        elif model_type == "power":
            return_dict = {
                "scale": fitted_params[0],
                "exponent": fitted_params[1],
                "nugget": fitted_params[2],
                "model_type": model_type,
            }
        else:
            return_dict = {
                "nugget": fitted_params[0],
                "sill": fitted_params[1],
                "range_": fitted_params[2],
                "model_type": model_type,
            }
        

        def variogram_model(lag):
            return model_function(lag, *fitted_params)        
        return_dict["variogram_model"] = variogram_model
        return return_dict
        
        

    def calculate_empirical_variogram(
        self,
        lag_step: float,
        angle: Optional[float],
        tolerance: float,
        max_range: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the variogram data for all frequency indices.

        Args:
            lag_step (float): The lag step size.
            angle (float, optional): The angle for the variogram.
            tolerance (float): The tolerance for the variogram.
            max_range (float, optional): The maximum range for the variogram.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The lags and semivariances for each frequency index.
        """
        all_semivariances = []
        all_lags = []
        for index in self.frequency_indices:
            lags, semivariances = self._calculate_variogram_data_at_index(
                lag_step, angle, tolerance, max_range, index
            )
            all_lags.append(lags)
            all_semivariances.append(semivariances)

        self.lags = np.array(all_lags)
        self.semivariances = np.array(all_semivariances)
        return self.lags, self.semivariances

    def _calculate_variogram_data_at_index(
        self,
        lag_step: float,
        angle: Optional[float],
        tolerance: float,
        max_range: Optional[float],
        index: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate the variogram data for a specific frequency index.

        Args:
            lag_step (float): The lag step size.
            angle (float, optional): The angle for the variogram.
            tolerance (float): The tolerance for the variogram.
            max_range (float, optional): The maximum range for the variogram.
            index (int): The frequency index.

        Returns:
            Tuple[List[float], List[float]]: The lags and semivariances for the frequency index.
        """
        lags = []
        semivariances = []
        current_lag = lag_step

        for semivariance in self.semivariances_generator_at_index(
            lag_step=lag_step,
            angle=angle,
            tolerance=tolerance,
            max_range=max_range,
            index=index,
        ):
            lags.append(current_lag)
            semivariances.append(semivariance)
            current_lag += lag_step

        return lags, semivariances

    def semivariances_generator_at_index(
        self,
        lag_step=1,
        angle: float | int | str = None,
        tolerance=5,
        max_range: int | float = None,
        increase_lag_after=None,
        increase_lag_by=3,
        weighted=False,
        deg=True,
        index=None,
    ):
        """
        Generate semivariances for a specific frequency index.

        Args:
            lag_step (float, optional): The lag step size. Defaults to 1.
            angle (float | int | str, optional): The angle for the variogram. Defaults to None.
            tolerance (float, optional): The tolerance for the variogram. Defaults to 5.
            max_range (int | float, optional): The maximum range for the variogram. Defaults to None.
            increase_lag_after (int, optional): The number of lags after which to increase the lag step. Defaults to None.
            increase_lag_by (int, optional): The factor by which to increase the lag step. Defaults to 3.
            weighted (bool, optional): Whether to use weighted semivariances. Defaults to False.
            deg (bool, optional): Whether the angle is in degrees. Defaults to True.
            index (int, optional): The frequency index. Defaults to None.

        Yields:
            float: The semivariance for each lag.
        """
        if isinstance(angle, str):
            if angle.capitalize() == "X":
                angle = 0
            elif angle.capitalize() == "Y":
                angle = 90 if deg else np.pi / 2
            else:
                raise ValueError("angle must be 'x', 'y' or a float")

        if deg and angle is not None:
            angle = np.deg2rad(angle)
            tolerance = np.deg2rad(tolerance)

        if self.position_grid is not None:
            if max_range is None:
                max_x = self.position_grid[0].max()
                max_y = self.position_grid[1].max()
                max_range_at_angle = max_x * np.cos(angle) + max_y * np.sin(angle)
            else:
                max_range_at_angle = max_range

            if increase_lag_after is False or increase_lag_after is None:
                num_lags_at_angle = max_range_at_angle / lag_step
                num_lags_at_angle = int(np.ceil(num_lags_at_angle))
            else:
                num_with_min_lag = max_range_at_angle / lag_step
                after_min_lag = num_with_min_lag - increase_lag_after
                num_with_increased_lag = after_min_lag / increase_lag_by
                num_lags_at_angle = increase_lag_after + num_with_increased_lag
                num_lags_at_angle = int(np.ceil(num_lags_at_angle))
        else:
            N, M = self.data.shape[:2]
            if max_range is None:
                max_range_at_angle = N * np.cos(angle) + M * np.sin(angle)
            else:
                max_range_at_angle = max_range

            if increase_lag_after is False or increase_lag_after is None:
                num_lags_at_angle = max_range_at_angle / lag_step
                num_lags_at_angle = int(np.ceil(num_lags_at_angle))
            else:
                num_with_min_lag = max_range_at_angle / lag_step
                after_min_lag = num_with_min_lag - increase_lag_after
                num_with_increased_lag = after_min_lag / increase_lag_by
                num_lags_at_angle = increase_lag_after + num_with_increased_lag
                num_lags_at_angle = int(np.ceil(num_lags_at_angle))

        for i in range(num_lags_at_angle):
            if increase_lag_after is not False and increase_lag_after is not None:
                if i < increase_lag_after:
                    lag_at_step = lag_step * (i + 1)
                    delta = (lag_step * 0.5, lag_step * 0.5)
                else:
                    lag_at_step = (
                        increase_lag_by * lag_step * (i + 1 - increase_lag_after)
                        + lag_step * increase_lag_after
                    )
                    delta = (increase_lag_by * lag_step * 0.5, increase_lag_by * lag_step * 0.5)
            else:
                lag_at_step = lag_step * (i + 1)
                delta = (lag_step * 0.5, lag_step * 0.5)
            semivariance = self.calculate_semivariance_at_index_and_lag(
                lag=lag_at_step,
                delta=delta,
                angle=angle,
                tolerance=tolerance,
                max_range=max_range,
                deg=False,
                weighted=weighted,
                index=index,
            )
            yield semivariance

    def calculate_semivariance_at_index_and_lag(
        self,
        lag,
        delta: Tuple[float, float] = None,
        angle=None,
        tolerance=5,
        consider_both_signs=True,
        max_range=False,
        deg=True,
        weighted=False,
        index=None,
    ):
        """
        Calculate the semivariance for a specific lag and frequency index.

        Args:
            lag (float): The lag distance.
            delta (Tuple[float, float], optional): The lag tolerance. Defaults to None.
            angle (float, optional): The angle for the variogram. Defaults to None.
            tolerance (float, optional): The tolerance for the variogram. Defaults to 5.
            consider_both_signs (bool, optional): Whether to consider both positive and negative angles. Defaults to True.
            max_range (float, optional): The maximum range for the variogram. Defaults to False.
            deg (bool, optional): Whether the angle is in degrees. Defaults to True.
            weighted (bool, optional): Whether to use weighted semivariances. Defaults to False.
            index (int, optional): The frequency index. Defaults to None.

        Returns:
            float: The semivariance for the specified lag and frequency index.
        """
        if delta is None:
            delta = (lag * 0.5, lag * 0.5)
        elif isinstance(delta, (int, float)):
            delta = (delta * 0.5, delta * 0.5)

        if self.position_grid is not None:
            x_coords = np.asarray(self.position_grid[0]).flatten()
            y_coords = np.asarray(self.position_grid[1]).flatten()
            X1, X2 = np.meshgrid(x_coords, x_coords)
            Y1, Y2 = np.meshgrid(y_coords, y_coords)
        else:
            X_ind, Y_ind = np.indices(self.data.shape[:2])
            x_coords = X_ind.flatten()
            y_coords = Y_ind.flatten()
            X1, X2 = np.meshgrid(x_coords, x_coords)
            Y1, Y2 = np.meshgrid(y_coords, y_coords)

        distances = np.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2)

        valid_pairs = (distances > (lag - delta[0])) & (distances <= (lag + delta[1]))

        if max_range:
            valid_pairs &= distances <= max_range

        if angle is not None:
            if deg:
                angle = np.deg2rad(angle)
                tolerance = np.deg2rad(tolerance)

            if not 0 <= angle < np.pi:
                raise ValueError("angle must be between 0 and 180 degrees")
            if not 0 <= tolerance < np.pi:
                raise ValueError("tolerance must be between 0 and 180 degrees")

            if angle == 0:
                angle = np.pi

            angles = np.arctan2(Y2 - Y1, X2 - X1)
            valid_angle_pairs = np.abs(angles - angle) <= tolerance
            if consider_both_signs:
                opposite_angle = angle - np.pi
                if opposite_angle == -np.pi:
                    opposite_angle = np.pi
                valid_angle_pairs |= np.abs(angles - opposite_angle) <= tolerance

            valid_pairs &= valid_angle_pairs

        values = self.data[..., index].flatten()
        D1, D2 = np.meshgrid(values, values)
        valid_semivariances = (D1[valid_pairs] - D2[valid_pairs]) ** 2

        if weighted:
            weights = (np.abs(D1[valid_pairs]) + np.abs(D2[valid_pairs])) / 2
            weighted_semivariances = valid_semivariances * weights
            return (
                np.average(weighted_semivariances, weights=weights)
                if weighted_semivariances.size > 0
                else 0
            )
        else:
            return np.mean(valid_semivariances) if valid_semivariances.size > 0 else 0

    def __repr__(self) -> str:
        """
        Return a string representation of the VariogramAnalyzer.

        Returns:
            str: The string representation of the VariogramAnalyzer.
        """
        return f"VariogramAnalyzer(data={self.data.shape}, position_grid={self.position_grid})"
