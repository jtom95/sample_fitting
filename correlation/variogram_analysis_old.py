from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from my_packages.auxiliary_plotting_functions.composite_plots import stack_figures
from my_packages.classes.aux_classes import Grid
from ._variogram_analysis_plotter_mixin import VariogramAnalyzerPlottingMixin


class VariogramAnalyzer(VariogramAnalyzerPlottingMixin):
    def __init__(
        self, data: np.ndarray, position_grid: Optional[Grid] = None, normalize: bool = True
    ):
        if normalize:
            data = self._normalize_data(data)
        self.data = data
        self.position_grid = position_grid
        self.df = pd.DataFrame(data.T)

    @staticmethod
    def _normalize_data(data: np.ndarray) -> np.ndarray:
        """
        Normalize the data to the range [0, 1].

        Args:
            data (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The normalized data.
        """
        return (data - data.min()) / (data.max() - data.min())
    
    def _calculate_variogram_data(
        self,
        lag_step: float,
        angle: Optional[float],
        tolerance: float,
        max_range: Optional[float],
        use_position_grid: bool,
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate the variogram data.

        Args:
            lag_step (float): The lag step size.
            angle (float, optional): The angle for the variogram.
            tolerance (float): The tolerance for the variogram.
            max_range (float, optional): The maximum range for the variogram.
            use_position_grid (bool): Whether to use the position grid.

        Returns:
            Tuple[List[float], List[float]]: The lags and semivariances for the variogram.
        """
        lags = []
        semivariances = []
        current_lag = lag_step

        for semivariance in self.semivariances_generator(
            lag_step=lag_step,
            angle=angle,
            tolerance=tolerance,
            max_range=max_range,
            use_grid=use_position_grid,
        ):
            lags.append(current_lag)
            semivariances.append(semivariance)
            current_lag += lag_step

        return lags, semivariances
    
    def semivariances_generator(
        self,
        lag_step=1,
        angle: float | int | str = None,
        tolerance=5,
        max_range: int | float = None,
        increase_lag_after=None,
        increase_lag_by=3,
        use_grid=False,
        weighted=False,
        deg=True,
    ):
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

        # the number of lags depends on the lag size, the angle and size of the grid
        if use_grid:
            if max_range is None:
                max_x = self.position_grid[0].max()
                max_y = self.position_grid[1].max()
                max_range_at_angle = max_x * np.cos(angle) + max_y * np.sin(angle)
            else:
                max_range_at_angle = max_range

            # x_step = self.position_grid[0][0, 1] - self.position_grid[0][0, 0]
            # y_step = self.position_grid[1][1, 0] - self.position_grid[1][0, 0]
            # step_at_angle = np.sqrt(x_step**2 + y_step**2) * np.cos(angle - np.pi / 4)
            if increase_lag_after is False or increase_lag_after is None:
                num_lags_at_angle = max_range_at_angle / lag_step
                num_lags_at_angle = int(np.ceil(num_lags_at_angle))
            else:
                # the first increase_lag_after lags are calculated with the given lag
                num_with_min_lag = max_range_at_angle / lag_step
                after_min_lag = num_with_min_lag - increase_lag_after
                # after the first increase_lag_after lags, the lag is set to increase_lag_by lag
                num_with_increased_lag = after_min_lag / increase_lag_by
                num_lags_at_angle = increase_lag_after + num_with_increased_lag
                num_lags_at_angle = int(np.ceil(num_lags_at_angle))

        else:
            N, M = self.data.shape
            if max_range is None:
                # max range is equal to N for angle 0 and to M for angle 90
                max_range_at_angle = N * np.cos(angle) + M * np.sin(angle)
            else:
                max_range_at_angle = max_range

            if increase_lag_after is False or increase_lag_after is None:
                num_lags_at_angle = max_range_at_angle / lag_step
                num_lags_at_angle = int(np.ceil(num_lags_at_angle))
            else:
                # the first increase_lag_after lags are calculated with the given lag
                num_with_min_lag = max_range_at_angle / lag_step
                after_min_lag = num_with_min_lag - increase_lag_after
                # after the first increase_lag_after lags, the lag is set to increase_lag_by lag
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
            semivariance = self.calculate_semivariance(
                lag=lag_at_step,
                delta=delta,
                angle=angle,
                tolerance=tolerance,
                max_range=max_range,
                use_grid=use_grid,
                deg=False,
                weighted=weighted,
            )
            yield semivariance

    def calculate_semivariance(
        self,
        lag,
        delta: Tuple[float, float] = None,
        angle=None,
        tolerance=5,
        consider_both_signs=True,
        max_range=False,
        use_grid=False,
        deg=True,
        weighted=False,
    ):
        """
        Calculates the semivariance for a given lag and angle. The semivariance is calculated as the
        mean of the squared differences between all points that are within the lag and angle range.
        Args:
            lag (int or float): This is the distance between the points that are compared.
            Any points that are within the lag range will be compared. The results are then averaged.
            angle (int or float): This is the angle between the points that are compared.
            Any points that are within the angle range will be compared. The results are then averaged.
            The angle is measured from the positive x-axis and is positive in the counter-clockwise direction.
            Also a tolerance is added to the angle range to allow for some variation in the angle.
            tolerance (int or float): This is the tolerance in the angle range. The tolerance is added to the
            angle range to allow for some variation in the angle.
            use_grid (bool, optional): If True, the ideal grid is used to calculate the semivariance.

        Returns:
            float: The semivariance for the given lag and angle.
        """

        if delta is None:
            delta = (lag * 0.5, lag * 0.5)
        elif isinstance(delta, (int, float)):
            delta = (delta * 0.5, delta * 0.5)

        if use_grid:
            x_coords = np.asarray(self.position_grid[0]).flatten()
            y_coords = np.asarray(self.position_grid[1]).flatten()
            # create a meshgrid of the x and y coordinates to get all possible combinations and speed up the calculation
            X1, X2 = np.meshgrid(x_coords, x_coords)
            Y1, Y2 = np.meshgrid(y_coords, y_coords)
        else:
            X_ind, Y_ind = np.indices(self.data.shape)
            x_coords = X_ind.flatten()
            y_coords = Y_ind.flatten()
            # create a meshgrid of the x and y coordinates to get all possible combinations and speed up the calculation
            X1, X2 = np.meshgrid(x_coords, x_coords)
            Y1, Y2 = np.meshgrid(y_coords, y_coords)

        # Calculate distances and angles between all pairs
        distances = np.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2)

        valid_pairs = (distances > (lag - delta[0])) & (distances <= (lag + delta[1]))

        if max_range:
            valid_pairs &= distances <= max_range

        if angle is not None:
            if deg:
                angle = np.deg2rad(angle)
                tolerance = np.deg2rad(tolerance)

            # set the angle and tolerance to be between 0 and pi
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

        values = self.data.flatten()
        D1, D2 = np.meshgrid(values, values)
        valid_semivariances = (D1[valid_pairs] - D2[valid_pairs]) ** 2

        if weighted:
            # Use the average value of the pair of points as weights
            weights = (np.abs(D1[valid_pairs]) + np.abs(D2[valid_pairs])) / 2
            weighted_semivariances = valid_semivariances * weights
            return (
                np.average(weighted_semivariances, weights=weights)
                if weighted_semivariances.size > 0
                else 0
            )
        else:
            # Unweighted semivariance
            return np.mean(valid_semivariances) if valid_semivariances.size > 0 else 0
    
