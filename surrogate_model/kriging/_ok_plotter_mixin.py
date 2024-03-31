from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from my_packages.EM_fields.scans import Scan, Grid
from process_scanner_measurements.sample_planes.sample_plane_operations import Transformations
from my_packages.EM_fields.plotting.fieldscans import plot_scans
from my_packages.constants import DistanceUnits

class OKPlotterMixinClass:
    def plot_dd_scatter(self, plots_per_row: int = 3, figsize: Tuple[int, int]=(8, 7), **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot the distance-distance scatter plot.
        """
        if self.variogram_analyzer is None:
            raise ValueError("Variogram analyzer has not been fitted yet. Call fit_variogram() first.")

        fig, ax = self.variogram_analyzer.plot_dd_scatter(
            plots_per_row=plots_per_row,
            figsize=figsize,
            units=self.configs.units,
            **kwargs
        )
        return fig, ax

    def plot_variogram(self, include_table: bool= True):
        """
        Plot the fitted variogram model.
        """
        if self.variogram_model is None:
            raise ValueError("Variogram model has not been fitted yet. Call fit_variogram() first.")

        lags = self.lags
        semivariances = self.semivariances

        model_function = self.variogram_model["model_function"]
        model_params = {k: v for k, v in self.variogram_model.items() if k in ("range_", "sill", "nugget", "slope", "exponent", "scale")}

        # rescale the model parameters
        for k, v in model_params.items():
            if k == "range_":
                model_params[k] = v * self.configs.units.value
            elif k == "sill":
                model_params[k] = v * self.label_scaler.scale_**2
            elif k == "nugget":
                model_params[k] = v * self.label_scaler.scale_**2
            elif k=="slope":
                model_params[k] = v * self.label_scaler.scale_**2
            elif k=="exponent":
                model_params[k] = v * self.label_scaler.scale_**2
            elif k=="scale":
                model_params[k] = v * self.label_scaler.scale_**2

        fitted_variogram_dict = {
            "model_type": self.variogram_model["model_type"],
            "model_function": self.variogram_model["model_function"],
            "variogram_generator": lambda x: model_function(x, **model_params),
        }
        fitted_variogram_dict.update(model_params)

        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.plot(lags, semivariances, "o", label="Empirical")

        # variogram_function = lambda x: model_function(x, **model_params)

        # lags_fine = np.linspace(0, np.max(lags), 100)
        # semivariances_fine = variogram_function(lags_fine)
        # ax.plot(lags_fine, semivariances_fine, "-", label=f"Fitted {self.variogram_model['model_type'].capitalize()} Model")

        # ax.xaxis.set_major_formatter(
        #     plt.FuncFormatter(lambda x, _: f"{x/self.configs.units.value:.1f}")
        # )

        # ax.set_xlabel(f"Distance {self.configs.units.name}")
        # ax.set_ylabel("Semivariance")
        # ax.set_title("Variogram Model")
        # ax.legend()

        fig, ax = self.variogram_analyzer.plot_fitted_variogram(fitted_variogram_dict, units = self.configs.units, include_table=include_table)

        return fig, ax

    def plot_prediction_comparison(self, grid: Grid, raw_scan: Scan):
        """
        Plot a comparison of the raw scan, OK prediction, and standard deviation.
        """
        fig, ax = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

        ok_scan, std = self.surrogate_model.predict_scan_and_std(grid, raw_scan.f)

        fig1, _ = plot_scans([raw_scan, ok_scan], ax=ax[:2])
        ax[0].set_title("Raw Scan")
        ax[1].set_title("GP Scan")
        fig1.suptitle(f"Frequency: {raw_scan.f*1e-6:.1f} MHz")

        cropped_std = std.crop_n_pixels(x=(10, 10), y=(10, 10))
        cropped_std.plot(ax=ax[2], cmap="hot", contour=True, levels=10)
        ax[2].set_title("Standard Deviation")

        return fig, ax

    def plot_prediction_error(self, X_test, y_test):
        """
        Plot the prediction error at test points.
        """
        if np.ndim(X_test) > 2:
            if X_test.shape[0] == 3: 
                X_test = X_test[:2]
            if X_test.shape[0] != 2:
                raise ValueError("X_test must be a 2D array with shape (n_samples, 2).")
            X_test = X_test.reshape(2, -1).T

        y_test = y_test.flatten()

        y_pred, std = self.predict(X_test, return_std=True)
        error = y_test - y_pred.squeeze()   

        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = max(abs(error.min()), abs(error.max()))
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=error, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Prediction Error")

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Error")

        return fig, ax

    def plot_prediction_surface(self, X_grid, cmap="viridis"):
        """
        Plot the prediction surface on a grid.
        """
        if np.ndim(X_grid) > 3:
            X_grid = X_grid[..., 0]

        y_pred = self.predict(X_grid)

        shape = X_grid.shape[1:]

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.contourf(
            X_grid[0],
            X_grid[1],
            y_pred.reshape(shape),
            cmap=cmap,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Prediction Surface")

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Predicted Value")

        return fig, ax
