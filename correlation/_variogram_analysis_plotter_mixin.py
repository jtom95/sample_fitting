from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


class VariogramAnalyzerPlottingMixin:
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
