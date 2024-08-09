from typing import List, Tuple, Union
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
        inlcude_samples=True,
        marker_size = 3,
        units = ""
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
            ax[0].imshow(
                Z,
                extent=(*xbounds, *ybounds),
                origin="lower",
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )

            ax[1].contourf(X, Y, std, cmap=cmap)
            for axx in ax:
                axx.set_xlabel("x")
                axx.set_ylabel("y")
            fig.colorbar(ax[0].images[0], ax=ax[0], label=units)
            fig.colorbar(ax[1].collections[0], ax=ax[1], label=units)

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

        if inlcude_samples:
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
            axx.set_xlabel("X [{}]".format(self._s_units.name))
            axx.set_ylabel("Y [{}]".format(self._s_units.name))
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
            self.plot_gp_2d(x, y, ax=ax[1:], include_std=include_std, inlcude_samples=include_samples, marker_size=marker_size, vmin=vmin, vmax=vmax, units = units)   
        else:
            self.plot_gp_2d(x, y, ax=ax[1], include_std=include_std, inlcude_samples=include_samples, marker_size=marker_size, vmin=vmin, vmax=vmax, units = units)    
        ax[0].set_title("Scan")
        ax[1].set_title("GP prediction")
        if include_std:
            ax[2].set_title("GP std")

        return fig, ax
