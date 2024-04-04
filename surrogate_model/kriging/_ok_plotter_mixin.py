from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


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

    def plot_variogram(self, include_table: bool= True, units: str="", ax=None) -> Tuple[plt.Figure, np.ndarray]:
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

        fig, ax = self.variogram_analyzer.plot_fitted_variogram(
            fitted_variogram_dict, 
            distance_units = self.configs.units, 
            units = units,
            include_table=include_table,
            ax = ax
            )

        return fig, ax

    def plot_prediction_comparison(
        self, grid: Grid, raw_scan: Scan, 
        units: str="", figsize:Tuple[int, int] = (10,3), 
        n_cropped_pixels: Tuple[int, int] = (0, 0),
        levels=10, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot a comparison of the raw scan, OK prediction, and standard deviation.
        """
        fig, ax = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

        ok_scan, std = self.surrogate_model.predict_scan_and_std(grid, raw_scan.f)
        
        vmax_raw = raw_scan.to_Scan().v.max()
        vmin_raw = raw_scan.to_Scan().v.min()
        
        vmax_ok = ok_scan.to_Scan().v.max()
        vmin_ok = ok_scan.to_Scan().v.min()
        
        vmax = max(vmax_raw, vmax_ok)
        vmin = min(vmin_raw, vmin_ok)
        
        raw_scan.plot(ax=ax[0], **kwargs, vmin=vmin, vmax=vmax, units=units, build_colorbar=False)
        ok_scan.plot(ax=ax[1], **kwargs, vmin=vmin, vmax=vmax, units=units)
        
        
        
        ax[0].set_title("Raw Scan")
        ax[1].set_title("Prediction Scan")
        fig.suptitle(f"Frequency: {raw_scan.f*1e-6:.1f} MHz")

        cropped_std = std.crop_n_pixels(x=n_cropped_pixels, y=n_cropped_pixels)
        cropped_std.plot(ax=ax[2], cmap="hot", contour=True, levels=levels, units=units)
        ax[2].set_title("Standard Deviation")

        return fig, ax

    def plot_prediction_error(self, X_test, y_test, figsize=(8, 6), units: str="", ax=None) -> Tuple[plt.Figure, np.ndarray]:
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
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        vmax = max(abs(error.min()), abs(error.max()))
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=error, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xlabel(f"X ({self.configs.units.name})")
        ax.set_ylabel(f"Y ({self.configs.units.name})")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/self.configs.units.value:.1f}")
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/self.configs.units.value:.1f}")
        )
        
        ax.set_title("Prediction Error")

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f"Error ({units})")

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

    def plot_kriging_weights(self, estimation_point, num_histogram_weights=10, figsize=(8,4), width_ratios=[3, 1]):
        """
        Plot the kriging weights for a given estimation point.

        Args:
            estimation_point (np.ndarray): The estimation point coordinates.
            ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot on. If None, a new figure and axes will be created.
            k (int, optional): The number of top weights to include in the histogram. Default is 10.

        Returns:
            tuple: A tuple containing the matplotlib figure and axes.
        """
        if self.kriging_estimator is None:
            raise ValueError("Kriging estimator has not been fitted yet. Call predict() first.")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=width_ratios)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        estimation_point = np.asarray(estimation_point)
        estimation_point_scaled = self.position_scaler.transform(estimation_point.reshape(1, -1)).flatten() / self.configs.units.value
        weights = self.kriging_estimator.calculate_weights(estimation_point_scaled)

        max_weights = np.max(np.abs(weights))

        q = ax1.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            c=weights,
            cmap="coolwarm",
            vmin=-max_weights,
            vmax=max_weights,
        )
        ax1.scatter(
            estimation_point[0],
            estimation_point[1],
            c="black",
            marker="x",
            s=100,
        )
        ax1.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/self.configs.units.value:.1f}")
        )
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/self.configs.units.value:.1f}")
        )
        ax1.set_xlabel(f"X ({self.configs.units.name})")
        ax1.set_ylabel(f"Y ({self.configs.units.name})")
        ax1.set_title("Kriging Weights")
        fig.colorbar(q, ax=ax1, label="Weight")
        
        k = min(num_histogram_weights, len(weights))

        top_k_indices = np.argsort(weights)[-k:]
        top_k_weights = weights[top_k_indices]
        top_k_positions = self.X_train[top_k_indices] / self.configs.units.value
        

        ax2.barh(np.arange(k), top_k_weights, align="center", alpha=0.5)
        ax2.set_yticks(np.arange(k))
        ax2.set_yticklabels([f"({pos[0]:.0f}, {pos[1]:.0f}) {self.configs.units.name}" for pos in top_k_positions])
        ax2.set_xlabel("Weight")
        ax2.set_title(f"Top {k} Weights")

        fig.tight_layout()

        return fig, (ax1, ax2)

    # def plot_kriging_weights(self, estimation_point, ax=None):
    #     """
    #     Plot the kriging weights for a given estimation point.

    #     Args:
    #         estimation_point (np.ndarray): The estimation point coordinates.
    #         ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot on. If None, a new figure and axes will be created.

    #     Returns:
    #         matplotlib.axes.Axes: The matplotlib axes containing the plot.
    #     """
    #     if self.kriging_estimator is None:
    #         raise ValueError("Kriging estimator has not been fitted yet. Call predict() first.")

    #     if ax is None:
    #         fig, ax = plt.subplots()

    #     estimation_point_scaled = np.asarray(estimation_point) / self.configs.units.value

    #     weights = self.kriging_estimator.calculate_weights(
    #         estimation_point_scaled,
    #     )

    #     q = ax.scatter(
    #         self.X_scaled[:, 0],
    #         self.X_scaled[:, 1],
    #         c=weights,
    #         cmap="coolwarm",
    #         vmin=-1,
    #         vmax=1,
    #     )
    #     ax.scatter(
    #         estimation_point_scaled[0],
    #         estimation_point_scaled[1],
    #         c="black",
    #         marker="x",
    #         s=100,
    #     )
    #     ax.set_xlabel(f"X ({self.configs.units.name})")
    #     ax.set_ylabel(f"Y ({self.configs.units.name})")
    #     ax.set_title("Kriging Weights")
    #     fig.colorbar(q, ax=ax, label="Weight")

    #     return ax
