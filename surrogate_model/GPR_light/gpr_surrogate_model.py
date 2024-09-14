from typing import List, Tuple, Union, Literal, Iterable
import numpy as np
import logging
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, Kernel
import matplotlib.pyplot as plt

from my_packages.EM_fields.scans import Scan, Grid
from my_packages.constants import DistanceUnits
from ..GPR._gaussian_regression_plotting import GaussianRegressionPlotterMixin

class GPR():
    MAX_DIM_RANGE = 0.1
    DEFAULT_KERNEL = C() * Matern()
    SPATIAL_UNITS_FOR_PLOTTING = DistanceUnits.mm
    def __init__(
        self, 
        kernel: Kernel = None,
        normalize_y: bool = True,
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 10,
        ):
        if kernel is None:
            kernel = self.DEFAULT_KERNEL

        self.kernel = kernel
        self.normalize_y = normalize_y
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer

        self.gp = GaussianProcessRegressor(
                kernel=kernel, 
                n_restarts_optimizer=n_restarts_optimizer, 
                alpha=alpha, 
                normalize_y=normalize_y
            )
        self.logger = logging.getLogger(__name__)
        self.current_acquisition_function = None

    @property
    def kernel_theta(self):
        if hasattr(self.gp, "kernel_"):
            return self.kernel.theta
    @property
    def _s_units(self):
        return self.SPATIAL_UNITS_FOR_PLOTTING

    def set_kernel_theta(self, theta: np.ndarray):
        self.kernel = self.kernel.clone_with_theta(theta)

    def get_initial_gp(self) -> GaussianProcessRegressor:
        return GaussianProcessRegressor(
            kernel=self.kernel, 
            n_restarts_optimizer=self.n_restarts_optimizer, 
            alpha=self.alpha, 
            normalize_y=self.normalize_y
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.gp.fit(X, y)

    def fit_Scan(self, scan: Scan):
        X = scan.grid.create_position_matrix()[:, :2]
        y = scan.v.flatten()
        self.gp.fit(X, y)

    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return self.gp.predict(X, return_std=return_std)

    def sample_prior_gp(self, X: np.ndarray, n_samples: int = 5, random_state=None) -> np.ndarray:
        gp = self.get_initial_gp()
        y_predicted = gp.sample_y(
            X, n_samples=n_samples, random_state=random_state
        )
        return y_predicted
    
    def sample_posterior_gp(self, X: np.ndarray, y: np.ndarray = None, n_samples: int = 5, random_state=None) -> np.ndarray:
        if y is not None:
            self.fit(X, y)
        y_predicted = self.gp.sample_y(
            X, n_samples=n_samples, random_state=random_state
        )
        return y_predicted

    def predict_grid(
        self, 
        x: np.ndarray,
        y: np.ndarray,
        return_std: bool = False
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # check that x and y are both 1D arrays
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        if return_std:
            values, std = self.predict(points, return_std=True)
            values, std = values.reshape(len(x), len(y)), std.reshape(len(x), len(y))
            return values, std
        values = self.predict(points)
        values = values.reshape(len(x), len(y))
        return values
    
    def predict_Scan(        
        self, 
        original_scan: Scan,
        x: np.ndarray = None,
        y: np.ndarray = None,
        return_std: bool = False
    )-> Union[Scan, Tuple[Scan, Scan]]:
        h = original_scan.height
        if x is None:
            x = original_scan.grid.x
        if y is None:
            y = original_scan.grid.y
            
        grid = Grid(np.meshgrid(x, y, h, indexing="ij"))
        if return_std:
            values, std = self.predict_grid(x, y, return_std=return_std)
            new_scan = original_scan.create_new(
                scan=values,
                grid=grid
            )   
            std = original_scan.create_new(
                scan=std,
                grid=grid,
            )
            return new_scan, std
        else:
            values = self.predict_grid(x, y)
            new_scan = original_scan.create_new(
                scan=values,
                grid=grid
            )
            return new_scan
        
    def plot_gp_2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ax=None,
        figsize=None,
        cmap="jet",
        vmin=None,
        vmax=None,
        include_std=True,
        include_samples=True,
        marker_size = 3,
        units = "",
        aspect = "auto"
    ):

        X, Y = np.meshgrid(x, y)
        xbounds = (x.min(), x.max())
        ybounds = (y.min(), y.max())

        if include_std:
            figsize = (10, 3) if figsize is None else figsize
            Z, std = self.gp.predict(np.array([X.ravel(), Y.ravel()]).T, return_std=True)
            Z, std = Z.reshape(X.shape), std.reshape(X.shape)
        else:
            figsize = (5, 3) if figsize is None else figsize
            Z = self.gp.predict(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)

        if ax is None:
            n_subplots = 2 if include_std else 1
            fig, ax = plt.subplots(1, n_subplots, figsize=figsize, constrained_layout=True)
        else:
            fig = ax[0].get_figure() if include_std else ax.get_figure()

        if include_std:
            q1 = ax[0].imshow(
                Z,
                extent=(*xbounds, *ybounds),
                origin="lower",
                cmap=cmap,
                aspect=aspect,
                vmin=vmin,
                vmax=vmax,
            )

            q2 = ax[1].contourf(X, Y, std, cmap=cmap)
            for axx in ax:
                axx.set_xlabel("x")
                axx.set_ylabel("y")
            fig.colorbar(q1, ax=ax[0], label=units)
            fig.colorbar(q2, ax=ax[1], label=units)

            ax[0].set_title("Prediction")
            ax[1].set_title("Standard deviation")
        else:
            ax.imshow(
                Z,
                extent=(*xbounds, *ybounds),
                origin="lower",
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(ax.images[0], ax=ax, label = units)

            ax.set_title("Prediction")

        if include_samples:
            training_X = self.gp.X_train_
            for axx in ax:
                axx.scatter(
                    training_X[:, 0],
                    training_X[:, 1],
                    color="w",
                    edgecolor="k",
                    s=marker_size,
                )
        for axx in ax:
            axx.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self._s_units.value:.1f}"))
            axx.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y/self._s_units.value:.1f}"))
            axx.set_xlabel("x [{}]".format(self._s_units.name))
            axx.set_ylabel("y [{}]".format(self._s_units.name))
        return fig, ax

    def plot_prior_gp_candidates(
        self, 
        X: np.ndarray, 
        n_samples: int = 4, 
        ax: plt.Axes = None, 
        linewidth: float = 1
    ):
        """
        Plot prior GP candidates.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param n_samples: Number of prior samples to draw.
        :param ax: Matplotlib axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        y_samples = self.sample_prior_gp(X, n_samples=n_samples)
        for i, y_sample in enumerate(y_samples.T):
            ax.plot(X[:, 0], y_sample, alpha=1, linewidth=linewidth)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self._s_units.value:.1f}"))
        ax.set_xlabel("X [{}]".format(self._s_units.name))        
        ax.set_ylabel("value")
        return fig, ax

    def plot_priors_on_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_samples: int = 4,
        ax=None,
        figsize=None,
        cmap="jet",
        vmin=None,
        vmax=None,
        units="",
        subplots_per_row=2,
        aspect="auto",
    ):
        X, Y = np.meshgrid(x, y)
        points = np.array([X.ravel(), Y.ravel()]).T

        y_samples = self.sample_prior_gp(points, n_samples=n_samples)

        n_subplots = (n_samples + subplots_per_row - 1) // subplots_per_row
        figsize = (5 * subplots_per_row, 5 * n_subplots) if figsize is None else figsize

        if ax is None:
            fig, ax = plt.subplots(
                n_subplots, subplots_per_row, figsize=figsize, constrained_layout=True
            )
            ax = ax.flatten()
        else:
            fig = ax[0].get_figure()

        for i in range(n_samples):
            Z = y_samples[:, i].reshape(X.shape)
            ax[i].imshow(
                Z,
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower",
                cmap=cmap,
                aspect=aspect,
                vmin=vmin,
                vmax=vmax,
            )
            ax[i].set_title(f"Sample {i + 1}")
            ax[i].set_xlabel(f"X [{self._s_units.name}]")
            ax[i].set_ylabel(f"Y [{self._s_units.name}]")
            fig.colorbar(ax[i].images[0], ax=ax[i], label=units)

        for j in range(i + 1, len(ax)):
            fig.delaxes(ax[j])

        return fig, ax
    
    def plot_posterior_gp_candidates(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 4,
        ax: plt.Axes = None,
        linewidth: float = 1,
        random_state: int = None,
    ):
        """
        Plot posterior GP candidates.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param y: 1D array of observed values (shape [n_samples]).
        :param n_samples: Number of posterior samples to draw.
        :param ax: Matplotlib axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        self.fit(X, y)
        y_samples = self.sample_posterior_gp(X, y, n_samples=n_samples, random_state=random_state)
        for i, y_sample in enumerate(y_samples.T):
            ax.plot(X[:, 0], y_sample, alpha=1, linewidth=linewidth)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self._s_units.value:.1f}"))
        ax.set_xlabel("X [{}]".format(self._s_units.name))
        ax.set_ylabel("value")
        return fig, ax
    
    def plot_posteriors_on_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_samples: int = 4,
        ax=None,
        figsize=None,
        cmap="jet",
        vmin=None,
        vmax=None,
        units="",
        subplots_per_row=2,
        aspect="auto",
        artistic = False,
        remove_cbars = False,
        include_samples = False,
        sample_scatter_size = 3,
        seed = None
    ):
        X, Y = np.meshgrid(x, y)
        points = np.array([X.ravel(), Y.ravel()]).T

        y_samples = self.sample_posterior_gp(points, n_samples=n_samples, random_state=seed)

        n_subplots = (n_samples + subplots_per_row - 1) // subplots_per_row
        figsize = (5 * subplots_per_row, 5 * n_subplots) if figsize is None else figsize

        if ax is None:
            fig, ax = plt.subplots(
                n_subplots, subplots_per_row, figsize=figsize, constrained_layout=True
            )
            ax = ax.flatten()
        else:
            fig = ax[0].get_figure()

        for i in range(n_samples):
            Z = y_samples[:, i].reshape(X.shape)
            ax[i].imshow(
                Z,
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower",
                cmap=cmap,
                aspect=aspect,
                vmin=vmin,
                vmax=vmax,
            )
            ax[i].set_title(f"Sample {i + 1}")
            ax[i].set_xlabel(f"X [{self._s_units.name}]")
            ax[i].set_ylabel(f"Y [{self._s_units.name}]")
            if not remove_cbars:
                fig.colorbar(ax[i].images[0], ax=ax[i], label=units)

        if include_samples:
            training_X = self.gp.X_train_
            for axx in ax:
                axx.scatter(
                    training_X[:, 0],
                    training_X[:, 1],
                    color="w",
                    edgecolor="k",
                    s=sample_scatter_size,
                )

        for j in range(i + 1, len(ax)):
            fig.delaxes(ax[j])

        if artistic:
            for axx in ax.flatten():
                axx.axis("off")
                axx.set_title("")
        return fig, ax

    def compare_with_scan(self, scan: Scan, ax=None, shape: Tuple[int, int] = (50, 50), include_std: bool = True, include_samples: bool = False, figsize=(12, 4), marker_size=3):
        if ax is None:
            n_plots = 3 if include_std else 2
            fig, ax = plt.subplots(1, n_plots, figsize=figsize, constrained_layout=True)
        else:
            fig = ax[0].get_figure()

        prediction = self.predict_grid(scan.grid.x, scan.grid.y, return_std=False)    

        vmax = np.max(np.concatenate([scan.v.flatten(), prediction.flatten()]))
        vmin = np.min(np.concatenate([scan.v.flatten(), prediction.flatten()]))
        units = "V/m" if scan.field_type.name == "E" else "A/m"

        scan.plot(ax=ax[0], vmin=vmin, vmax=vmax, build_colorbar=False)
        new_shape_x, new_shape_y = shape
        x = np.linspace(scan.grid.x.min(), scan.grid.x.max(), new_shape_x)
        y = np.linspace(scan.grid.y.min(), scan.grid.y.max(), new_shape_y)
        if include_std:
            self.plot_gp_2d(x, y, ax=ax[1:], include_std=include_std, include_samples=include_samples, marker_size=marker_size, vmin=vmin, vmax=vmax, units = units)   
        else:
            self.plot_gp_2d(x, y, ax=ax[1], include_std=include_std, include_samples=include_samples, marker_size=marker_size, vmin=vmin, vmax=vmax, units = units)    
        ax[0].set_title("Scan")
        ax[1].set_title("GP prediction")
        if include_std:
            ax[2].set_title("GP std")

        return fig, ax
    
    
    def plot_kernel_1d(self, ax=None, figsize=(6, 4), n_points=100):
        """
        Plot the kernel along the x and y axes.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()
        
        x_range = self._calculate_default_range(axis="x")
        y_range = self._calculate_default_range(axis="y")
        
        overall_range  = max(x_range, y_range)
        overall_range = overall_range * 1.5
        
        x = np.linspace(0, overall_range, n_points)
        y = np.linspace(0, overall_range, n_points)
        
        points_x = np.array([x, np.zeros_like(x)]).T
        points_y = np.array([np.zeros_like(y), y]).T
        
        Z_x = self._evaluate_kernel_for_lag(points_x).squeeze()
        Z_y = self._evaluate_kernel_for_lag(points_y).squeeze()
        
        ax.plot(x, Z_x, label="x")
        ax.plot(y, Z_y, label="y")
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self._s_units.value:.1f}"))
        ax.set_xlabel(f"Lag Distance [{self._s_units.name}]")
        ax.set_ylabel("Kernel value")
        ax.legend()
        ax.grid()  
        return fig, ax
        
    
    def plot_kernel_along_axis(self, ax=None, axis: Literal["x", "y"] = "y", figsize=(6, 4), n_points=100, label=None):
        """
        Plot the kernel along the x or y axis.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        if axis == "x":
            x = np.linspace(0, self._calculate_default_range(axis="x")*1.5, n_points)
            y = np.zeros_like(x)
        elif axis == "y":
            y = np.linspace(0, self._calculate_default_range(axis="y")*1.5, n_points)
            x = np.zeros_like(y)

        points = np.array([x, y]).T
        Z = self._evaluate_kernel_for_lag(points).squeeze()
        ax.plot(x, Z, label=label)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self._s_units.value:.1f}"))
        ax.set_xlabel(f"{axis.upper()} [{self._s_units.name}]")
        ax.set_ylabel("Kernel value")
        return fig, ax

    def plot_kernel_heatmap(self, ax=None, cmap="jet", figsize=(6, 4), xres=100, yres=100):
        """
        Plot the fitted kernel. If the kernel is anisotropic, plot both axes.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        if hasattr(self.gp, "kernel_"):
            kernel = self.gp.kernel_
        else:
            kernel = self.kernel

        x_range = self._calculate_default_range(axis="x")
        y_range = self._calculate_default_range(axis="y")
        
        drange = max(x_range, y_range)

        x = np.linspace(0, drange, xres)
        y = np.linspace(0, drange, yres)
        X, Y = np.meshgrid(x, y)
        grid_points = np.array([X.ravel(), Y.ravel()]).reshape(2, -1).T
        Z = self._evaluate_kernel_for_lag(grid_points).reshape(xres, yres)
        c = ax.imshow(Z, origin="lower", extent=(0, drange, 0, drange), cmap=cmap)
        fig.colorbar(c, ax=ax, label="Kernel value")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self._s_units.value:.1f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y/self._s_units.value:.1f}"))
        ax.set_xlabel("X [{}]".format(self._s_units.name))
        ax.set_ylabel("Y [{}]".format(self._s_units.name))
        return fig, ax


    def _evaluate_kernel_for_lag(self, lag: Iterable[float]):
        position_0 = np.array([[0, 0]])
        if not isinstance(lag, np.ndarray):
            lag = np.array(lag)
        if np.ndim(lag) == 1:
            lag = lag.reshape(1, -1)
        elif np.ndim(lag) > 2:
            raise ValueError("lag must be a 1D or 2D array")
        if lag.shape[1] == 1:
            lag = np.concatenate([lag, np.zeros((lag.shape[0], 1))], axis=1)
        elif lag.shape[1] > 2:
            raise ValueError("lag must have 1 or 2 columns") 
        return self.gp.kernel_(position_0, lag)

    def _calculate_default_range(
        self,
        axis: Literal["x", "y"] = "y",
        ratio: float = 1 / 20,
        max_iterations: int = 20,
        tolerance: float = 1e-4,
    ):
        max_field = float(self._evaluate_kernel_for_lag([0, 0]).squeeze())
        target_field = max_field * ratio

        left, right = 0, self.MAX_DIM_RANGE
        for _ in range(max_iterations):
            mid = (left + right) / 2
            if axis == "x":
                mid_point = [[mid, 0]]
            elif axis == "y":
                mid_point = [[0, mid]]
            else:
                raise ValueError("axis must be 'x' or 'y'")

            field_at_mid = float(self._evaluate_kernel_for_lag(mid_point).squeeze())
            if np.abs(field_at_mid - target_field) < tolerance:
                break
            elif field_at_mid > target_field:
                left = mid
            else:
                right = mid

        return mid
