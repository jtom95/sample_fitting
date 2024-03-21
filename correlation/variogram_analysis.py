from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from my_packages.auxiliary_plotting_functions.composite_plots import stack_figures
from my_packages.classes.aux_classes import Grid


class VariogramAnalyzer:
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

    def plot_xy_variograms(
        self,
        lag_step_x: float = 1,
        lag_step_y: float = 1,
        max_range_x: Optional[float] = None,
        max_range_y: Optional[float] = None,
        tolerance_x: float = 5,
        tolerance_y: float = 5,
        label_x: str = "x",
        label_y: str = "y",
        use_position_grid: bool = False,
        rotated_imshow: bool = False,
        ax: Optional[plt.Axes] = None,
        aspect: str = "auto",
        **kwargs,
    ) -> plt.Figure:
        """
        Plot the variograms along the x and y directions.

        Args:
            lag_step_x (float, optional): The lag step size in the x direction. Defaults to 1.
            lag_step_y (float, optional): The lag step size in the y direction. Defaults to 1.
            max_range_x (float, optional): The maximum range in the x direction. Defaults to None.
            max_range_y (float, optional): The maximum range in the y direction. Defaults to None.
            tolerance_x (float, optional): The tolerance in the x direction. Defaults to 5.
            tolerance_y (float, optional): The tolerance in the y direction. Defaults to 5.
            label_x (str, optional): The label for the x variogram. Defaults to "x".
            label_y (str, optional): The label for the y variogram. Defaults to "y".
            use_position_grid (bool, optional): Whether to use the position grid. Defaults to False.
            rotated_imshow (bool, optional): Whether to rotate the imshow plot. Defaults to False.
            ax (plt.Axes, optional): The axes to plot on. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the imshow function.

        Returns:
            plt.Figure: The figure containing the variogram plots.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True)

        # Prepare the data and extent for imshow
        data_to_show, extent = self._prepare_imshow_data(use_position_grid, rotated_imshow)

        # Plot the data using imshow
        self._plot_imshow(
            ax[0], data_to_show, extent, use_position_grid, rotated_imshow, aspect=aspect, **kwargs
        )

        # Plot the variograms
        self._plot_variogram(
            ax[1],
            lag_step=lag_step_x,
            angle="x",
            max_range=max_range_x,
            tolerance=tolerance_x,
            label=label_x,
            use_position_grid=use_position_grid,
        )
        self._plot_variogram(
            ax[1],
            lag_step=lag_step_y,
            angle="y",
            max_range=max_range_y,
            tolerance=tolerance_y,
            label=label_y,
            use_position_grid=use_position_grid,
        )

        ax[0].set_title("Scan")
        ax[1].set_title("Variogram")
        ax[1].legend()

        fig.suptitle("XY Variograms")
        return fig

    def _prepare_imshow_data(
        self, use_position_grid: bool, rotated_imshow: bool
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Prepare the data and extent for the imshow plot.

        Args:
            use_position_grid (bool): Whether to use the position grid.
            rotated_imshow (bool): Whether to rotate the imshow plot.

        Returns:
            Tuple[np.ndarray, Tuple[float, float, float, float]]: The data to show and the extent for imshow.
        """
        if use_position_grid:
            if rotated_imshow:
                extent = [
                    self.position_grid[1].min(),
                    self.position_grid[1].max(),
                    self.position_grid[0].min(),
                    self.position_grid[0].max(),
                ]
            else:
                extent = [
                    self.position_grid[0].min(),
                    self.position_grid[0].max(),
                    self.position_grid[1].min(),
                    self.position_grid[1].max(),
                ]
        else:
            if rotated_imshow:
                extent = [0, self.data.shape[1], 0, self.data.shape[0]]
            else:
                extent = [0, self.data.shape[0], 0, self.data.shape[1]]

        data_to_show = self.data.T[::-1]

        return data_to_show, extent

    def _plot_imshow(
        self,
        ax: plt.Axes,
        data_to_show: np.ndarray,
        extent: Tuple[float, float, float, float],
        use_position_grid: bool,
        rotated_imshow: bool,
        aspect: str = "auto",
        **kwargs,
    ) -> None:
        """
        Plot the data using imshow.

        Args:
            ax (plt.Axes): The axes to plot on.
            data_to_show (np.ndarray): The data to show.
            extent (Tuple[float, float, float, float]): The extent for imshow.
            use_position_grid (bool): Whether to use the position grid.
            rotated_imshow (bool): Whether to rotate the imshow plot.
            **kwargs: Additional keyword arguments to pass to the imshow function.
        """
        if rotated_imshow:
            q = ax.imshow(data_to_show.T, origin="lower", extent=extent, aspect=aspect, **kwargs)
        else:
            q = ax.imshow(data_to_show, origin="lower", extent=extent, aspect=aspect, **kwargs)

        if use_position_grid:
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
            if rotated_imshow:
                ax.set_ylabel("X [mm]")
                ax.set_xlabel("Y [mm]")
            else:
                ax.set_ylabel("Y [mm]")
                ax.set_xlabel("X [mm]")
        else:
            if rotated_imshow:
                ax.set_ylabel("X index")
                ax.set_xlabel("Y index")
            else:
                ax.set_ylabel("Y index")
                ax.set_xlabel("X index")

        plt.colorbar(q, ax=ax, label="dBm")

    def _plot_variogram(
        self,
        ax: plt.Axes,
        lag_step: float = 1,
        angle: Optional[float] = None,
        tolerance: float = 5,
        max_range: Optional[float] = None,
        label: str = "",
        use_position_grid: bool = False,
    ) -> None:
        """
        Plot the variogram on the given axes.

        Args:
            ax (plt.Axes): The axes to plot on.
            lag_step (float, optional): The lag step size. Defaults to 1.
            angle (float, optional): The angle for the variogram. Defaults to None.
            tolerance (float, optional): The tolerance for the variogram. Defaults to 5.
            max_range (float, optional): The maximum range for the variogram. Defaults to None.
            label (str, optional): The label for the variogram. Defaults to "".
            use_position_grid (bool, optional): Whether to use the position grid. Defaults to False.
        """
        lags, semivariances = self._calculate_variogram_data(
            lag_step, angle, tolerance, max_range, use_position_grid
        )

        ax.plot(lags, semivariances, marker="o", label=label)
        ax.set_xlabel("Lag Distance [mm]" if use_position_grid else "Lag Distance")
        ax.set_ylabel("Semivariance")
        ax.grid(True)

        if use_position_grid:
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")

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

    def plot_variogram(
        self,
        lag_step=1,
        angle=None,
        tolerance=5,
        ax: plt.Axes = None,
        increase_lag_after=None,
        increase_lag_by=3,
        max_range=None,
        use_grid=False,
        weighted=False,
        deg=True,
        **kwargs,
    ):
        """
        Plots the variogram for the given parameters.

        Args:
            lag (int, optional): The starting lag distance. Defaults to 1.
            angle (float | int | str, optional): The angle for directional variogram. Defaults to "x".
            tolerance (float, optional): The angular tolerance in degrees. Defaults to 0.5.
            use_grid (bool, optional): Whether to use an ideal grid. Defaults to False.
            deg (bool, optional): Whether the angle is in degrees. Defaults to False.
        """

        # Generate semivariances and lags
        semivariances = []
        lags = []
        current_lag = lag_step
        for semivariance in self.semivariances_generator(
            lag_step=lag_step,
            angle=angle,
            tolerance=tolerance,
            max_range=max_range,
            increase_lag_after=increase_lag_after,
            increase_lag_by=increase_lag_by,
            use_grid=use_grid,
            weighted=weighted,
            deg=deg,
        ):
            semivariances.append(semivariance)
            lags.append(current_lag)
            if increase_lag_after is False or increase_lag_after is None:
                current_lag += lag_step
            else:
                if len(lags) < increase_lag_after:
                    current_lag += lag_step
                else:
                    current_lag += lag_step * increase_lag_by
        # Plotting the variogram
        if ax is None:
            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        ## kwargs for the plot
        default_kwargs = dict(
            marker="o",
        )

        kwargs = {**default_kwargs, **kwargs}

        ax.plot(lags, semivariances, **kwargs)
        ax.set_title(f"Variogram - Angle: {angle}")
        if use_grid:
            ax.set_xlabel("Lag Distance [mm]")
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
        ax.set_ylabel("Semivariance")
        ax.grid(True)
        return fig

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

    def plot_points_with_distance_condition(
        self, lag, delta, point_coords="central", ax=None, use_grid=False, **kwargs
    ):
        """
        Plots the points that satisfy the distance condition relative to the central point.
        """
        if point_coords == "central":
            point_coords = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        if use_grid:
            central_point = (
                self.position_grid[0][point_coords[0], point_coords[1]],
                self.position_grid[1][point_coords[0], point_coords[1]],
            )
        else:
            central_point = point_coords

        if use_grid:
            x_coords, y_coords = (
                np.asarray(self.position_grid[0]).flatten(),
                np.asarray(self.position_grid[1]).flatten(),
            )
        else:
            x_coords, y_coords = np.indices(self.data.shape)
            x_coords, y_coords = x_coords.flatten(), y_coords.flatten()

        distances = np.sqrt((x_coords - central_point[0]) ** 2 + (y_coords - central_point[1]) ** 2)

        valid_points = (distances > (lag - 0.5 * delta)) & (distances <= (lag + 0.5 * delta))

        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        def_kwargs = dict(
            marker="o",
            label="Valid Points",
            c="red",
        )

        kwargs = {**def_kwargs, **kwargs}

        # if color is passed in kwargs remove the c argument
        if "color" in kwargs:
            kwargs.pop("c")

        ax.scatter(x_coords[valid_points], y_coords[valid_points], **kwargs)
        ax.scatter(central_point[0], central_point[1], c="k", marker="x")
        ax.set_title(f"Lag Distance {lag}")

        if use_grid:
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
            xlims = (self.position_grid[0].min(), self.position_grid[0].max())
            ylims = (self.position_grid[1].min(), self.position_grid[1].max())

        else:
            ax.set_xlabel("X index")
            ax.set_ylabel("Y index")
            xlims = (0, self.data.shape[0])
            ylims = (0, self.data.shape[1])

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.legend()
        ax.grid(True)

        return fig

    def plot_points_with_angle_condition(
        self,
        angle,
        tolerance,
        point_coords="central",
        ax=None,
        consider_both_signs=True,
        use_grid=False,
        deg=True,
        **kwargs,
    ):
        """
        Plots the points that satisfy the distance condition relative to the central point.
        """
        if point_coords == "central":
            point_coords = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        if use_grid:
            central_point = (
                self.position_grid[0][point_coords[0], point_coords[1]],
                self.position_grid[1][point_coords[0], point_coords[1]],
            )
        else:
            central_point = point_coords

        if use_grid:
            x_coords, y_coords = (
                np.asarray(self.position_grid[0]).flatten(),
                np.asarray(self.position_grid[1]).flatten(),
            )
        else:
            x_coords, y_coords = np.indices(self.data.shape)
            x_coords, y_coords = x_coords.flatten(), y_coords.flatten()

        angles = np.arctan2(y_coords - central_point[1], x_coords - central_point[0])

        if deg:
            angle = np.deg2rad(angle)
            tolerance = np.deg2rad(tolerance)

        valid_points = np.abs(angles - angle) <= tolerance

        # consider also the opposite angle
        if consider_both_signs:
            #     opposite_angle = (
            #         angle - np.pi
            #     )  # the input angle is between 0 and 2pi and the opposite angle is between -pi and pi
            #     if opposite_angle == -np.pi:
            #         opposite_angle = np.pi
            valid_points |= np.abs(angles - angle - np.pi) <= tolerance
            valid_points |= np.abs(angles - angle + np.pi) <= tolerance

        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        def_kwargs = dict(
            marker="o",
            label="Valid Points",
            c="red",
        )

        kwargs = {**def_kwargs, **kwargs}

        # if color is passed in kwargs remove the c argument
        if "color" in kwargs:
            kwargs.pop("c")

        ax.scatter(x_coords[valid_points], y_coords[valid_points], **kwargs)

        if use_grid:
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
            xlims = (self.position_grid[0].min(), self.position_grid[0].max())
            ylims = (self.position_grid[1].min(), self.position_grid[1].max())

        else:
            ax.set_xlabel("X index")
            ax.set_ylabel("Y index")
            xlims = (0, self.data.shape[0])
            ylims = (0, self.data.shape[1])

        ax.scatter(central_point[0], central_point[1], c="k", marker="x")
        ax.set_title(f"Angle {np.rad2deg(angle):.0f}° ± {np.rad2deg(tolerance):.0f}°")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.legend()
        ax.grid(True)
        return fig

    def plot_lag_and_angle_conditions(
        self,
        lag=1,
        angle="x",
        tolerance=5,
        delta=1,
        point_coords="central",
        both_signs=True,
        ax=None,
        use_grid=False,
        deg=True,
        **kwargs,
    ):
        if isinstance(angle, str):
            if angle.capitalize() == "X":
                angle = 0
            elif angle.capitalize() == "Y":
                angle = 90 if deg else np.pi / 2
            else:
                raise ValueError("angle must be 'x', 'y' or a float")

        if deg:
            angle = np.deg2rad(angle)
            tolerance = np.deg2rad(tolerance)

        if use_grid:
            x_coords, y_coords = (
                np.asarray(self.position_grid[0]).flatten(),
                np.asarray(self.position_grid[1]).flatten(),
            )
        else:
            x_coords, y_coords = np.indices(self.data.shape)
            x_coords, y_coords = x_coords.flatten(), y_coords.flatten()

        if point_coords == "central":
            point_coords = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        if use_grid:
            central_point = (
                self.position_grid[0][point_coords[0], point_coords[1]],
                self.position_grid[1][point_coords[0], point_coords[1]],
            )
        else:
            central_point = point_coords

        # Calculate distances and angles between all pairs
        distances = np.sqrt((x_coords - central_point[0]) ** 2 + (y_coords - central_point[1]) ** 2)
        angles = np.arctan2(y_coords - central_point[1], x_coords - central_point[0])

        valid_points = (distances > (lag - 0.5 * delta)) & (distances <= (lag + 0.5 * delta))
        angle_valid_points = np.abs(angles - angle) <= tolerance
        if both_signs:
            opposite_angle = angle - np.pi
            if opposite_angle == -np.pi:
                opposite_angle = np.pi
            angle_valid_points |= np.abs(angles - opposite_angle) <= tolerance

        valid_points &= angle_valid_points

        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        def_kwargs = dict(
            marker="o",
            label="Valid Points",
        )

        kwargs = {**def_kwargs, **kwargs}

        ax.scatter(x_coords[valid_points], y_coords[valid_points], **kwargs)
        ax.scatter(central_point[0], central_point[1], c="blue", label="Central Point")
        ax.set_title(f"Lag {lag} @ {np.rad2deg(angle):.0f}° ± {np.rad2deg(tolerance):.0f}°")
        ax.set_xlim(0, self.data.shape[0])
        ax.set_ylim(0, self.data.shape[1])
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        ax.grid(True)
        return fig
