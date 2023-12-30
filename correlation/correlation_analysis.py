from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from my_packages.auxiliary_plotting_functions.composite_plots import stack_figures
from my_packages.classes.aux_classes import Grid


class CorrelationAnalyzer:
    def __init__(self, data: np.ndarray, ideal_grid: Grid = None):
        self.data = data
        self.ideal_grid = ideal_grid
        self.df = pd.DataFrame(data.T)

    def get_correlation_matrix(self, axis: int = 0) -> np.ndarray:
        if axis == 0:
            return self.df.corr()
        elif axis == 1:
            return self.df.T.corr()
        else:
            raise ValueError("axis must be 0 or 1")

    def get_mean_matrix(self, axis: int = 0) -> np.ndarray:
        mean_axis = self.df.max(axis=axis).to_numpy()

        N = len(mean_axis)
        mean_matrix = np.zeros((N, N))
        for ii in range(N):
            for jj in range(N):
                mean_matrix[ii, jj] = np.mean([mean_axis[ii], mean_axis[jj]])
        return mean_matrix

    def plot_xy_variograms(
        self,
        lag_stepx=1,
        lag_stepy=1,
        max_rangex=None,
        max_rangey=None,
        tolerancex=5,
        tolerancey=5,
        label_x="x",
        label_y="y",
        use_ideal_grid=False,
        ax=None,
        **kwargs,
    ):
        fig, ax = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True)
        q = ax[0].imshow(
            self.data, origin="lower", extent=[0, self.data.shape[1], 0, self.data.shape[0]]
        )
        plt.colorbar(q, ax=ax[0], label="dBm")
        self.plot_variogram(
            lag_step=lag_stepx,
            angle="x",
            max_range=max_rangex,
            tolerance=tolerancex,
            ax=ax[1],
            label=label_x,
            use_ideal_grid=use_ideal_grid,
        )
        self.plot_variogram(
            lag_step=lag_stepy,
            angle="y",
            max_range=max_rangey,
            tolerance=tolerancey,
            ax=ax[1],
            label=label_y,
            use_ideal_grid=use_ideal_grid,
        )
        ax[0].set_title("Scan")
        ax[1].set_title("Variogram")
        ax[1].legend()
        
        fig.suptitle("XY Variograms")
        return fig

    def plot_variogram(
        self,
        lag_step=1,
        angle=None,
        tolerance=5,
        ax: plt.Axes = None,
        increase_lag_after=None,
        increase_lag_by=3,
        max_range=None,
        use_ideal_grid=False,
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
            use_ideal_grid (bool, optional): Whether to use an ideal grid. Defaults to False.
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
            use_ideal_grid=use_ideal_grid,
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
        if use_ideal_grid:
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
        use_ideal_grid=False,
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
        if use_ideal_grid:
            if max_range is None:
                max_x = self.ideal_grid[0].max()
                max_y = self.ideal_grid[1].max()
                max_range_at_angle = max_x * np.cos(angle) + max_y * np.sin(angle)
            else:
                max_range_at_angle = max_range

            # x_step = self.ideal_grid[0][0, 1] - self.ideal_grid[0][0, 0]
            # y_step = self.ideal_grid[1][1, 0] - self.ideal_grid[1][0, 0]
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
                use_ideal_grid=use_ideal_grid,
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
        use_ideal_grid=False,
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
            use_ideal_grid (bool, optional): If True, the ideal grid is used to calculate the semivariance.

        Returns:
            float: The semivariance for the given lag and angle.
        """

        if delta is None:
            delta = (lag * 0.5, lag * 0.5)
        elif isinstance(delta, (int, float)):
            delta = (delta * 0.5, delta * 0.5)

        if use_ideal_grid:
            x_coords = np.asarray(self.ideal_grid[0]).flatten()
            y_coords = np.asarray(self.ideal_grid[1]).flatten()
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
        self, lag, delta, point_coords="central", ax=None, use_ideal_grid=False
    ):
        """
        Plots the points that satisfy the distance condition relative to the central point.
        """
        if point_coords == "central":
            point_coords = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        if use_ideal_grid:
            central_point = (
                self.ideal_grid[0][point_coords[0], point_coords[1]],
                self.ideal_grid[1][point_coords[0], point_coords[1]],
            )
        else:
            central_point = point_coords

        if use_ideal_grid:
            x_coords, y_coords = (
                np.asarray(self.ideal_grid[0]).flatten(),
                np.asarray(self.ideal_grid[1]).flatten(),
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

        ax.scatter(x_coords[valid_points], y_coords[valid_points], c="red", label="Valid Points")
        ax.scatter(central_point[0], central_point[1], c="blue", label="Central Point")
        ax.set_title(f"Lag Distance {lag}")
        ax.set_xlim(0, self.data.shape[0])
        ax.set_ylim(0, self.data.shape[1])
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
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
        use_ideal_grid=False,
        deg=True,
        **kwargs,
    ):
        """
        Plots the points that satisfy the distance condition relative to the central point.
        """
        if point_coords == "central":
            point_coords = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        if use_ideal_grid:
            central_point = (
                self.ideal_grid[0][point_coords[0], point_coords[1]],
                self.ideal_grid[1][point_coords[0], point_coords[1]],
            )
        else:
            central_point = point_coords

        if use_ideal_grid:
            x_coords, y_coords = (
                np.asarray(self.ideal_grid[0]).flatten(),
                np.asarray(self.ideal_grid[1]).flatten(),
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
            opposite_angle = (
                angle - np.pi
            )  # the input angle is between 0 and 2pi and the opposite angle is between -pi and pi
            if opposite_angle == -np.pi:
                opposite_angle = np.pi
            valid_points |= np.abs(angles - opposite_angle) <= tolerance

        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        ax.scatter(
            x_coords[valid_points], y_coords[valid_points], c="red", label="Valid Points", **kwargs
        )
        ax.scatter(central_point[0], central_point[1], c="blue", label="Central Point")
        ax.set_title(f"Angle {np.rad2deg(angle):.0f}° ± {np.rad2deg(tolerance):.0f}°")
        ax.set_xlim(0, self.data.shape[0])
        ax.set_ylim(0, self.data.shape[1])
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
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
        use_ideal_grid=False,
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

        if use_ideal_grid:
            x_coords, y_coords = (
                np.asarray(self.ideal_grid[0]).flatten(),
                np.asarray(self.ideal_grid[1]).flatten(),
            )
        else:
            x_coords, y_coords = np.indices(self.data.shape)
            x_coords, y_coords = x_coords.flatten(), y_coords.flatten()

        if point_coords == "central":
            point_coords = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        if use_ideal_grid:
            central_point = (
                self.ideal_grid[0][point_coords[0], point_coords[1]],
                self.ideal_grid[1][point_coords[0], point_coords[1]],
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

    def plot_correlation(
        self,
        apply_variable_alpha: bool = True,
        units: str = "dBm",
        heatmap_title: str = "Scan Heatmap",
        cmap_heatmap: str = "jet",
        cmap_correlation: str = "hot",
        figsize_heatmap: tuple = (10, 3),
        figsize_correlation: tuple = (10, 5),
    ):
        if apply_variable_alpha:
            mean_x_matrix = self.get_mean_matrix(axis=0)
            mean_y_matrix = self.get_mean_matrix(axis=1)
            # normalize the mean matrices between 0 and 1
            alpha_x = (mean_x_matrix - mean_x_matrix.min()) / (
                mean_x_matrix.max() - mean_x_matrix.min()
            )
            alpha_y = (mean_y_matrix - mean_y_matrix.min()) / (
                mean_y_matrix.max() - mean_y_matrix.min()
            )
        else:
            alpha_x = 1
            alpha_y = 1
        # get the correlation matrices
        correlation_matrix_X = self.get_correlation_matrix(axis=0)
        correlation_matrix_Y = self.get_correlation_matrix(axis=1)

        ## plot the correlation matrices
        fig_plot, ax = plt.subplots(figsize=figsize_heatmap, constrained_layout=True)
        q = ax.imshow(self.data, aspect="auto", cmap=cmap_heatmap)
        plt.colorbar(ax=ax, mappable=q, label=f"[{units}]")
        ax.set_xlabel("y_index")
        ax.set_ylabel("x_index")
        ax.set_title(heatmap_title, fontsize=16)

        fig, ax = plt.subplots(2, 2, figsize=figsize_correlation, constrained_layout=True)

        ax[0, 0].plot(self.df.T)
        ax[0, 1].plot(self.df)
        q_row = ax[1, 0].imshow(
            correlation_matrix_X, aspect="auto", cmap=cmap_correlation, alpha=alpha_x
        )
        q_column = ax[1, 1].imshow(
            correlation_matrix_Y, aspect="auto", cmap=cmap_correlation, alpha=alpha_y
        )

        ax[0, 0].set_title("Signals Along X")
        ax[0, 1].set_title("Signals Along Y")
        ax[0, 0].set_xlabel("y_index")
        ax[0, 0].set_ylabel(f"[{units}]")
        ax[0, 1].set_xlabel("x_index")
        ax[0, 1].set_ylabel(f"[{units}]")

        ax[1, 0].set_title("correlation along X axis")
        ax[1, 1].set_title("correlation along Y axis")
        ax[1, 0].set_xlabel("x_index")
        ax[1, 1].set_xlabel("y_index")

        ax[1, 0].set_ylabel("x_index")
        ax[1, 1].set_ylabel("y_index")

        # for axx in ax.flatten():
        #     xmin, xmax = axx.get_xlim()
        #     ymin, ymax = axx.get_ylim()
        #     axx.set_xticks(list(axx.get_xticks()) + [xmax-0.5])
        #     axx.set_yticks(list(axx.get_yticks()) + [ymax-0.5])
        #     axx.set_xlim(xmin, xmax)
        #     axx.set_ylim(ymin, ymax)

        # add colorbars
        fig.colorbar(q_row, ax=ax[1, 0])
        fig.colorbar(q_column, ax=ax[1, 1])

        ffig, _ = stack_figures([fig_plot, fig], height_ratios=[0.5, 1], aspect="equal", top=0.95)
        ffig.suptitle("Correlation Analysis - No Grid", fontsize=14)

        return ffig
