from typing import Optional, Tuple, Callable, Dict
import numpy as np
import matplotlib.pyplot as plt
from .variogram_models import VariogramModels
from my_packages.constants import DistanceUnits     

class VariogramAnalyzerPlottingMixin:

    def plot_fitted_variogram(
        self,
        variogram_dict: Dict,
        ax: plt.Axes = None,
        distance_units: DistanceUnits = DistanceUnits.mm,
        include_table: bool = True,
        units: str = "",
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the fitted variogram.

        Args:
            ax (plt.Axes, optional): The axes to plot on. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes containing the variogram plot.
        """
        if self.lags is None or self.semivariances is None:
            raise ValueError(
                "Variogram data has not been calculated. Please call calculate_empirical_variogram first."
            )

        fitted_values = variogram_dict

        model_type = fitted_values.get("model_type")
        variogram_model = fitted_values.get("variogram_generator")

        distance_vals = np.linspace(0, self.lags[-1], 100)

        variogram_fitted_vals = variogram_model(distance_vals)

        if ax is None:
            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        default_kwargs = dict(marker="x", linestyle=" ")
        kwargs = {**default_kwargs, **kwargs}

        ax.plot(self.lags, self.semivariances, label="empirical", **kwargs)
        ax.plot(distance_vals, variogram_fitted_vals, label=model_type+" model")
        ax.set_title(f"Fitted Variogram")
        ax.set_xlabel(f"Lag Distance [{distance_units.name}]")
        ylabel = "Semivariance"
        if units != "":
            ylabel += f" [{units}]"
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

        # change the x-axis to the distance_units
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/distance_units.value:.1f}")
        )
        # Add model parameters as text
        model_params_str = ""
        model_params = {
            k: param
            for k, param in fitted_values.items()
            if k not in ["model_type", "variogram_generator", "model_function", "residuals"]
        }

        for k, param in model_params.items():
            if k == "range_":
                k = "Range"
                if isinstance(param, (list, np.ndarray)):
                    param = param[0]
                param = param / distance_units.value
                model_params_str += f"{k}: {param:.1f} {distance_units.name}\n"
                continue
            else:
                k = k.capitalize()
            if isinstance(param, (int, float)):
                model_params_str += f"{k}: {param:.3f}\n"
            elif isinstance(param, (list, np.ndarray)):
                model_params_str += f"{k}: {', '.join([f'{p:.3f}' for p in param])}\n"
            else:
                model_params_str += f"{k}: {param}\n"
        if include_table:
            ax.text(
                0.95, 0.05, model_params_str, transform=ax.transAxes, 
                ha="right", va="bottom", 
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5')
                )

        return fig, ax

    def plot_empirical_variogram(
        self,
        ax: plt.Axes = None,
        distance_units: DistanceUnits = DistanceUnits.mm,
        units: str = "",
        **kwargs,
    ) -> plt.Figure:
        """
        Plot the empirical variogram.

        Args:
            ax (plt.Axes, optional): The axes to plot on. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            plt.Figure: The figure containing the variogram plot.
        """
        if isinstance(distance_units, str):
            distance_units = DistanceUnits[distance_units]
        
        if self.lags is None or self.semivariances is None:
            raise ValueError(
                "Variogram data has not been calculated. Please call calculate_empirical_variogram first."
            )

        if ax is None:
            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        default_kwargs = dict(marker="o", linestyle=" ")
        kwargs = {**default_kwargs, **kwargs}

        ax.plot(self.lags, self.semivariances, **kwargs)
        ax.set_title(f"Empirical Variogram")
        ax.set_xlabel(f"Lag Distance [{distance_units.name}]")
        ylabel = "Semivariance"
        if units != "":
            ylabel += f" [{units}]"
        ax.set_ylabel(ylabel)
        ax.grid(True)

        # change the x-axis to the distance_units
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/distance_units.value:.1f}'))
        return fig

    def plot_correlogram(self, ax=None, distance_units: DistanceUnits = DistanceUnits.mm, **kwargs):
        if self.lags is None or self.semivariances is None:
            raise ValueError("Empirical variogram data not available. Call calculate_empirical_variogram() first.")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Calculate the variance (C_x(0))
        variance = np.var(self.values)

        # Calculate the covariance (C_x(d)) using the relationship between semivariance and covariance
        covariance = variance - self.semivariances

        # Calculate the correlogram (rho_x(d))
        correlogram = covariance / variance

        default_kwargs = dict(marker='o', linestyle=' ')
        kwargs = {**default_kwargs, **kwargs}

        ax.plot(self.lags, correlogram, **kwargs)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/distance_units.value:.1f}'))
        ax.set_xlabel(f'Lag Distance [{distance_units.name}]')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlogram')
        ax.grid(True)

        return fig

    def plot_dd_scatter(
        self,
        plots_per_row: int = 3,
        figsize: Tuple[int, int] = (8, 5),
        units: str = "",
        distance_units: DistanceUnits = DistanceUnits.mm,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        lags = self.lags
        angles = self.angles_
        tolerances = self.tolerances_
        deltas = self.deltas_

        if lags is None or angles is None or tolerances is None or deltas is None:
            raise ValueError(
                "Empirical variogram data not available. Call calculate_empirical_variogram() first."
            )

        n = len(lags)
        nrows = n // plots_per_row + (
            0 if n % plots_per_row == 0 else 1
        )  # Adjusted to correctly calculate the number of rows
        ncols = min(n, plots_per_row)

        fig, ax = plt.subplots(
            nrows, ncols, figsize=figsize, constrained_layout=True, sharex=True, sharey=True
        )

        for i, lag in enumerate(lags):
            row = i // ncols
            col = i % ncols
            self.plot_dd_scatter_at_lag(
                lag,
                deltas[i],
                angle=angles[i],
                angle_tolerance=tolerances[i],
                ax=ax[row, col],
                distance_units=distance_units,
                units = units,
                **kwargs,
            )
            ax[row, col].set_title(f"Lag: {lag/distance_units.value:.1f} {distance_units.name}")

        # Remove empty axes
        for i in range(n, nrows * ncols):
            row = i // ncols
            col = i % ncols
            fig.delaxes(ax[row, col])

        # Adjust label removal logic
        # Remove the x labels from the plots that are not in the last row
        for i in range(nrows - 1):
            for j in range(ncols):
                ax[i, j].set_xlabel("")

        for i in range(nrows):
            for j in range(1, ncols):
                ax[i, j].set_ylabel("")

        return fig, ax

    def plot_dd_scatter_at_lag(
        self, lag, delta, 
        angle=0, angle_tolerance=90, 
        ax=None, aspect="equal", 
        units: str = "",
        distance_units: DistanceUnits = DistanceUnits.mm,
        **kwargs)-> Tuple[plt.Figure, plt.Axes]:
        if isinstance(delta, (int, float)):
            delta = np.array([delta, delta]) * 0.5

        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            fig = ax.figure

        # Filter pairs by lag distance and angle
        valid_pairs = self.get_pairs_at_lag(
            lag, delta=delta, angle=angle, tolerance=angle_tolerance, deg=True, register_parameters=False
        )

        D1, D2 = np.meshgrid(self.values, self.values)
        head_values = D1[valid_pairs]
        tail_values = D2[valid_pairs]

        default_kwargs = dict(marker="o", alpha=0.5)
        kwargs = {**default_kwargs, **kwargs}

        ax.scatter(head_values, tail_values, **kwargs)
        xlabel = f"Head Value"
        ylabel = f"Tail Value"

        if units != "":
            xlabel += f" [{units}]"
            ylabel += f" [{units}]"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        title = ""
        if lag is not None:
            title += f"Lag {lag/distance_units.value} ± {delta[0]/distance_units.value} {distance_units.name}"
        if angle is not None:
            if title:
                title += " @ "
            title += f"{angle:.0f}° ± {angle_tolerance:.0f}°"

        ax.set_title(title)

        # set the aspect ratio
        ax.set_aspect(aspect)

        return fig, ax

    def plot_dd_scatter_at_lags(self, lags, delta, angle=None, angle_tolerance=90, ncols=2, figsize=(8, 6), **kwargs):
        n = len(lags)
        nrows = n // ncols + 1

        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)

        for i, lag in enumerate(lags):
            row = i // ncols
            col = i % ncols
            self.plot_dd_scatter(lag, delta, angle=angle, angle_tolerance=angle_tolerance, ax=ax[row, col], **kwargs)

        # remove empty axes
        for i in range(n, nrows * ncols):
            row = i // ncols
            col = i % ncols
            fig.delaxes(ax[row, col])

    def plot_points_with_conditions(
        self,
        lag=None,
        delta=None,
        angle=None,
        tolerance=None,
        point_coords="central",
        consider_both_signs=True,
        ax=None,
        deg=True,
        distance_units: DistanceUnits = DistanceUnits.mm,
        **kwargs,
    ):
        if isinstance(distance_units, str):
            distance_units = DistanceUnits[distance_units]
        central_point = self._get_central_point(point_coords)
        distances = np.sqrt(np.sum((self.positions - central_point) ** 2, axis=1))
        angles = np.arctan2(
            self.positions[:, 1] - central_point[1], self.positions[:, 0] - central_point[0]
        )

        valid_points = np.ones(distances.shape, dtype=bool)

        if lag is not None:
            if isinstance(delta, (int, float)):
                delta = np.array([delta, delta]) * 0.5
            valid_points &= (distances > (lag - delta[0])) & (distances <= (lag + delta[1]))

        if angle is not None:
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

            angle_valid_points = np.abs(angles - angle) <= tolerance

            if consider_both_signs:
                opposite_angle = angle - np.pi
                if opposite_angle == -np.pi:
                    opposite_angle = np.pi
                angle_valid_points |= np.abs(angles - opposite_angle) <= tolerance

            valid_points &= angle_valid_points

        fig, ax = self._setup_plot(ax)
        self._plot_points(ax, valid_points, central_point, **kwargs)

        title = ""
        if lag is not None:
            title += f"Lag {lag/distance_units.value} ± {delta[0]/distance_units.value} {distance_units.name}"
        if angle is not None:
            if title:
                title += " @ "
            title += f"{np.rad2deg(angle):.0f}° ± {np.rad2deg(tolerance):.0f}°"
        ax.set_title(title)

        self._format_plot(ax, distance_units)

        return fig

    def _get_central_point(self, point_coords):
        if point_coords == "central":
            mean_value = np.mean(self.positions, axis=0)
            # find the closest point to the mean value
            central_point = self.positions[np.argmin(np.sum((self.positions - mean_value) ** 2, axis=1))]
            return central_point
        else:
            x, y = point_coords
            # find the closest point to the given coordinates
            central_point = self.positions[np.argmin(np.sum((self.positions[:, :2] - np.array([x, y])) ** 2, axis=1))]
            return central_point

    def _setup_plot(self, ax):
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        return fig, ax

    def _plot_points(self, ax, valid_points, central_point, **kwargs):
        def_kwargs = dict(
            marker="o",
            label="Valid Points",
            c="red",
        )
        kwargs = {**def_kwargs, **kwargs}
        if "color" in kwargs:
            kwargs.pop("c")

        ax.scatter(self.positions[valid_points, 0], self.positions[valid_points, 1], **kwargs)
        ax.scatter(central_point[0], central_point[1], c="k", marker="x")

    def _format_plot(self, ax, distance_units):
        xlims = (self.positions[:, 0].min(), self.positions[:, 0].max())
        ylims = (self.positions[:, 1].min(), self.positions[:, 1].max())
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/distance_units.value:.1f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/distance_units.value:.1f}'))

        ax.set_xlabel(f"X Coordinate [{distance_units.name}]")
        ax.set_ylabel(f"Y Coordinate [{distance_units.name}]")
        ax.legend()
        ax.grid(True)
