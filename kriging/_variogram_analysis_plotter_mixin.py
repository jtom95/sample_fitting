from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from .variogram_models import VariogramModels

class VariogramAnalyzerPlottingMixin:

    def plot_fitted_variogram_at_index(
        self, 
        index: int,
        model_type: str,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot the fitted variogram for a specific frequency index.

        Args:
            index (int): The frequency index.
            ax (plt.Axes, optional): The axes to plot on. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            plt.Figure: The figure containing the variogram plot.
        """
        if self.lags is None or self.semivariances is None:
            raise ValueError("Variogram data has not been calculated. Please call calculate_variogram_data first.")

        fitted_values: dict = self.fit_variogram_model_at_index(
            idx=index,
            model_type=model_type,
        )

        model_type = fitted_values.pop("model_type")
        variogram_model = fitted_values.pop("variogram_model")

        position = self.frequency_indices_order_dict[index]
        distance_vals = np.linspace(0, self.lags[position, -1], 100)

        variogram_fitted_vals = variogram_model(distance_vals)
        
        if ax is None:
            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        default_kwargs = dict(marker="x", linestyle=" ")
        kwargs = {**default_kwargs, **kwargs}

        index_position = self.frequency_indices_order_dict[index]

        ax.plot(self.lags[index_position], self.semivariances[index_position], label="empirical", **kwargs)
        ax.plot(distance_vals, variogram_fitted_vals, label="fitted "+model_type)
        ax.set_title(f"Fitted Variogram - Frequency Index {index}")
        ax.set_xlabel("Lag Distance [mm]" if self.position_grid is not None else "Lag Distance")
        ax.set_ylabel("Semivariance")
        ax.legend()
        ax.grid(True)

        if self.position_grid is not None:
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")

        return fig

    def plot_empirical_variogram_at_index(
        self,
        index: int,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot the variogram for a specific frequency index.

        Args:
            index (int): The frequency index.
            lag_step (float, optional): The lag step size. Defaults to 1.
            angle (float, optional): The angle for the variogram. Defaults to None.
            tolerance (float, optional): The tolerance for the variogram. Defaults to 5.
            ax (plt.Axes, optional): The axes to plot on. Defaults to None.
            increase_lag_after (int, optional): The number of lags after which to increase the lag step. Defaults to None.
            increase_lag_by (int, optional): The factor by which to increase the lag step. Defaults to 3.
            max_range (float, optional): The maximum range for the variogram. Defaults to None.
            weighted (bool, optional): Whether to use weighted semivariances. Defaults to False.
            deg (bool, optional): Whether the angle is in degrees. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            plt.Figure: The figure containing the variogram plot.
        """
        if self.lags is None or self.semivariances is None:
            raise ValueError("Variogram data has not been calculated. Please call calculate_variogram_data first.")

        if ax is None:
            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        default_kwargs = dict(marker="o", linestyle=" ")
        kwargs = {**default_kwargs, **kwargs}

        index_position = self.frequency_indices_order_dict[index]

        ax.plot(self.lags[index_position], self.semivariances[index_position], **kwargs)
        ax.set_title(f"Variogram - Frequency Index {index}")
        ax.set_xlabel("Lag Distance [mm]" if self.position_grid is not None else "Lag Distance")
        ax.set_ylabel("Semivariance")
        ax.grid(True)

        if self.position_grid is not None:
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")

        return fig

    def plot_empirical_variograms(
        self,
        plots_per_row: int = 3,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot the variograms for all frequency indices.

        Args:
            lag_step (float, optional): The lag step size. Defaults to 1.
            angle (float, optional): The angle for the variogram. Defaults to None.
            tolerance (float, optional): The tolerance for the variogram. Defaults to 5.
            increase_lag_after (int, optional): The number of lags after which to increase the lag step. Defaults to None.
            increase_lag_by (int, optional): The factor by which to increase the lag step. Defaults to 3.
            max_range (float, optional): The maximum range for the variogram. Defaults to None.
            weighted (bool, optional): Whether to use weighted semivariances. Defaults to False.
            deg (bool, optional): Whether the angle is in degrees. Defaults to True.
            plots_per_row (int, optional): The number of subplots per row. Defaults to 3.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            plt.Figure: The figure containing the variogram plots for all frequency indices.
        """
        num_indices = len(self.frequency_indices)
        num_rows = int(np.ceil(num_indices / plots_per_row))
        num_cols = min(plots_per_row, num_indices)

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows), constrained_layout=True
        )
        for i, index in enumerate(self.frequency_indices):
            row = i // plots_per_row
            col = i % plots_per_row

            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]

            self.plot_empirical_variogram_at_index(
                    ax=ax,
                    index=index,
                    **kwargs          
                )

        # Remove any unused subplots
        for i in range(num_indices, num_rows * num_cols):
            row = i // plots_per_row
            col = i % plots_per_row

            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]

            ax.remove()

        fig.suptitle("Variograms - All Frequency Indices")
        return fig

    def plot_points_with_distance_condition(
        self, lag, delta, point_coords="central", ax=None, **kwargs
    ):
        """
        Plot the points that satisfy the distance condition relative to the central point.
        """
        if point_coords == "central":
            point_coords = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        if self.position_grid is not None:
            central_point = (
                self.position_grid[0][point_coords[0], point_coords[1]],
                self.position_grid[1][point_coords[0], point_coords[1]],
            )
        else:
            central_point = point_coords

        if self.position_grid is not None:
            x_coords, y_coords = (
                np.asarray(self.position_grid[0]).flatten(),
                np.asarray(self.position_grid[1]).flatten(),
            )
        else:
            x_coords, y_coords = np.indices(self.data.shape[:2])
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

        if self.position_grid is not None:
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
        deg=True,
        **kwargs,
    ):
        """
        Plot the points that satisfy the angle condition relative to the central point.
        """
        if point_coords == "central":
            point_coords = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        if self.position_grid is not None:
            central_point = (
                self.position_grid[0][point_coords[0], point_coords[1]],
                self.position_grid[1][point_coords[0], point_coords[1]],
            )
        else:
            central_point = point_coords

        if self.position_grid is not None:
            x_coords, y_coords = (
                np.asarray(self.position_grid[0]).flatten(),
                np.asarray(self.position_grid[1]).flatten(),
            )
        else:
            x_coords, y_coords = np.indices(self.data.shape[:2])
            x_coords, y_coords = x_coords.flatten(), y_coords.flatten()

        angles = np.arctan2(y_coords - central_point[1], x_coords - central_point[0])

        if deg:
            angle = np.deg2rad(angle)
            tolerance = np.deg2rad(tolerance)

        valid_points = np.abs(angles - angle) <= tolerance

        # consider also the opposite angle
        if consider_both_signs:
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

        if self.position_grid is not None:
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

        if self.position_grid is not None:
            x_coords, y_coords = (
                np.asarray(self.position_grid[0]).flatten(),
                np.asarray(self.position_grid[1]).flatten(),
            )
        else:
            x_coords, y_coords = np.indices(self.data.shape[:2])
            x_coords, y_coords = x_coords.flatten(), y_coords.flatten()

        if point_coords == "central":
            point_coords = (self.data.shape[0] // 2, self.data.shape[1] // 2)
        if self.position_grid is not None:
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


######################################################################
######################################################################

class OtherPlottingFuncs:

    def plot_xy_variograms_at_index(
        self,
        index: int,
        lag_step_x: float = 1,
        lag_step_y: float = 1,
        max_range_x: Optional[float] = None,
        max_range_y: Optional[float] = None,
        tolerance_x: float = 5,
        tolerance_y: float = 5,
        label_x: str = "x",
        label_y: str = "y",
        rotated_imshow: bool = False,
        ax: Optional[plt.Axes] = None,
        aspect: str = "auto",
        **kwargs,
    ) -> plt.Figure:
        """
        Plot the variograms along the x and y directions for a specific frequency index.

        Args:
            index (int): The frequency index to plot.
            lag_step_x (float, optional): The lag step size in the x direction. Defaults to 1.
            lag_step_y (float, optional): The lag step size in the y direction. Defaults to 1.
            max_range_x (float, optional): The maximum range in the x direction. Defaults to None.
            max_range_y (float, optional): The maximum range in the y direction. Defaults to None.
            tolerance_x (float, optional): The tolerance in the x direction. Defaults to 5.
            tolerance_y (float, optional): The tolerance in the y direction. Defaults to 5.
            label_x (str, optional): The label for the x variogram. Defaults to "x".
            label_y (str, optional): The label for the y variogram. Defaults to "y".
            rotated_imshow (bool, optional): Whether to rotate the imshow plot. Defaults to False.
            ax (plt.Axes, optional): The axes to plot on. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the imshow function.

        Returns:
            plt.Figure: The figure containing the variogram plots.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True)

        # Prepare the data and extent for imshow
        data_to_show, extent = self._prepare_imshow_data(rotated_imshow)

        # Plot the data using imshow
        self._plot_imshow(
            ax[0], data_to_show[..., index], extent, rotated_imshow, aspect=aspect, **kwargs
        )

        # Plot the variograms
        self._plot_variogram_at_index(
            ax[1],
            index=index,
            lag_step=lag_step_x,
            angle="x",
            max_range=max_range_x,
            tolerance=tolerance_x,
            label=label_x,
        )
        self._plot_variogram_at_index(
            ax[1],
            index=index,
            lag_step=lag_step_y,
            angle="y",
            max_range=max_range_y,
            tolerance=tolerance_y,
            label=label_y,
        )

        ax[0].set_title(f"Scan - Frequency Index {index}")
        ax[1].set_title(f"Variogram - Frequency Index {index}")
        ax[1].legend()

        fig.suptitle(f"XY Variograms - Frequency Index {index}")
        return fig

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
        rotated_imshow: bool = False,
        plots_per_row: int = 3,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot the variograms along the x and y directions for all frequency indices.

        Args:
            lag_step_x (float, optional): The lag step size in the x direction. Defaults to 1.
            lag_step_y (float, optional): The lag step size in the y direction. Defaults to 1.
            max_range_x (float, optional): The maximum range in the x direction. Defaults to None.
            max_range_y (float, optional): The maximum range in the y direction. Defaults to None.
            tolerance_x (float, optional): The tolerance in the x direction. Defaults to 5.
            tolerance_y (float, optional): The tolerance in the y direction. Defaults to 5.
            label_x (str, optional): The label for the x variogram. Defaults to "x".
            label_y (str, optional): The label for the y variogram. Defaults to "y".
            rotated_imshow (bool, optional): Whether to rotate the imshow plot. Defaults to False.
            plots_per_row (int, optional): The number of subplots per row. Defaults to 3.
            **kwargs: Additional keyword arguments to pass to the imshow function.

        Returns:
            plt.Figure: The figure containing the variogram plots for all frequency indices.
        """
        num_indices = len(self.frequency_indices)
        num_rows = int(np.ceil(num_indices / plots_per_row))
        num_cols = min(plots_per_row, num_indices)

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows), constrained_layout=True
        )

        for i, index in enumerate(self.frequency_indices):
            row = i // plots_per_row
            col = i % plots_per_row

            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]

            self.plot_xy_variograms_at_index(
                index=index,
                lag_step_x=lag_step_x,
                lag_step_y=lag_step_y,
                max_range_x=max_range_x,
                max_range_y=max_range_y,
                tolerance_x=tolerance_x,
                tolerance_y=tolerance_y,
                label_x=label_x,
                label_y=label_y,
                rotated_imshow=rotated_imshow,
                ax=ax,
                **kwargs,
            )

        # Remove any unused subplots
        for i in range(num_indices, num_rows * num_cols):
            row = i // plots_per_row
            col = i % plots_per_row

            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]

            ax.remove()

        fig.suptitle("XY Variograms - All Frequency Indices")
        return fig

    def _prepare_imshow_data(
        self, rotated_imshow: bool
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Prepare the data and extent for the imshow plot.

        Args:
            rotated_imshow (bool): Whether to rotate the imshow plot.

        Returns:
            Tuple[np.ndarray, Tuple[float, float, float, float]]: The data to show and the extent for imshow.
        """
        if self.position_grid is not None:
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
            rotated_imshow (bool): Whether to rotate the imshow plot.
            **kwargs: Additional keyword arguments to pass to the imshow function.
        """
        if rotated_imshow:
            q = ax.imshow(data_to_show.T, origin="lower", extent=extent, aspect=aspect, **kwargs)
        else:
            q = ax.imshow(data_to_show, origin="lower", extent=extent, aspect=aspect, **kwargs)

        if self.position_grid is not None:
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

    def _plot_variogram_at_index(
        self,
        ax: plt.Axes,
        index: int,
        lag_step: float = 1,
        angle: Optional[float] = None,
        tolerance: float = 5,
        max_range: Optional[float] = None,
        label: str = "",
    ) -> None:
        """
        Plot the variogram on the given axes for a specific frequency index.

        Args:
            ax (plt.Axes): The axes to plot on.
            index (int): The frequency index.
            lag_step (float, optional): The lag step size. Defaults to 1.
            angle (float, optional): The angle for the variogram. Defaults to None.
            tolerance (float, optional): The tolerance for the variogram. Defaults to 5.
            max_range (float, optional): The maximum range for the variogram. Defaults to None.
            label (str, optional): The label for the variogram. Defaults to "".
        """
        lags, semivariances = self._calculate_variogram_data_at_index(
            lag_step, angle, tolerance, max_range, index
        )

        ax.plot(lags, semivariances, marker="o", label=label)
        ax.set_xlabel("Lag Distance [mm]" if self.position_grid is not None else "Lag Distance")
        ax.set_ylabel("Semivariance")
        ax.grid(True)

        if self.position_grid is not None:
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*1e3:.0f}")
