import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

from .kernel_manager import Kernel


class GaussianRegressionPlotterMixin:
    def plot_prior_gp_mean_and_std(self, X: np.ndarray, ax: plt.Axes = None):
        """
        Plot prior GP candidates.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param n_samples: Number of prior samples to draw.
        :param ax: Matplotlib axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        y_mean, y_std = self.predict_prior_gp(X, return_std=True)
        ax.plot(X[:, 0], y_mean, color="black", label="Mean")
        ax.fill_between(
            X[:, 0], y_mean - y_std, y_mean + y_std, color="gray", alpha=0.2, label="Std"
        )
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self.units.value:.1f}"))
        ax.set_xlabel("X [{}]".format(self.units.name))
        ax.set_ylabel("y")
        ax.legend()

    def plot_prior_gp_candidates(
        self, X: np.ndarray, n_samples: int = 4, ax: plt.Axes = None, linewidth: float = 1
    ):
        """
        Plot prior GP candidates.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param n_samples: Number of prior samples to draw.
        :param ax: Matplotlib axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        y_samples = self.sample_prior_gp(X, n_samples=n_samples)
        for i, y_sample in enumerate(y_samples.T):
            ax.plot(X[:, 0], y_sample, alpha=1, linewidth=linewidth)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self.units.value:.1f}"))
        ax.set_xlabel("X [{}]".format(self.units.name))        
        ax.set_ylabel("y")
    
    def plot_prior_gp_candidates_with_theoretical_stats(
        self,
        X: np.ndarray,
        n_samples: int = 4,
        ax: plt.Axes = None,
        candidate_linewidth: float = 0.5,
        figsize: Tuple[int, int] = (8, 6),
        random_state = None,
    ):
        """
        Plot prior GP candidates.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param n_samples: Number of prior samples to draw.
        :param ax: Matplotlib axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
            
        self.plot_prior_gp_mean_and_std(X, ax=ax)
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
            
        y_samples = self.sample_prior_gp(X, n_samples=n_samples, random_state=random_state)
        for i, y_sample in enumerate(y_samples.T):
            ax.plot(X[:, 0], y_sample, alpha=1, linestyle="-", linewidth=candidate_linewidth)
            
        return fig, ax
        

    def plot_prior_gp_candidates_with_sample_stats(
        self,
        X: np.ndarray,
        n_samples: int = 4,
        ax: plt.Axes = None,
        random_state: int = None,
        candidate_linewidth: float = 0.5,
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
        y_samples = self.sample_prior_gp(X, n_samples=n_samples, random_state=random_state)
        for i, y_sample in enumerate(y_samples.T):
            ax.plot(X[:, 0], y_sample, alpha=1, linestyle="-", linewidth=candidate_linewidth)

        dataframe = pd.DataFrame(y_samples, columns=[f"Sample {i+1}" for i in range(n_samples)])
        sample_std = dataframe.std(axis=1)
        sample_mean = dataframe.mean(axis=1)

        ax.fill_between(
            X[:, 0],
            sample_mean - sample_std,
            sample_mean + sample_std,
            color="gray",
            alpha=0.5,
            label="Std",
        )

        ax.plot(X[:, 0], sample_mean, color="black", label="Mean")

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self.units.value:.1f}"))
        ax.set_xlabel("X [{}]".format(self.units.name))
        ax.set_ylabel("y")
        ax.legend()
        return fig, ax

    def plot_prior_gp_2d(
        self,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        cmap: str = "jet",
        n_samples: int = 4,
        figs_per_row: int = 2,
        figsize: Tuple[int, int] = (5, 5),
        include_colorbar: bool = True,
        random_state = None,
        artistic: bool = False,
    ):
        if not np.ndim(x_axis) == 1 or not np.ndim(y_axis) == 1:
            raise ValueError("x_axis and y_axis must be 1D arrays")

        grid = np.array(np.meshgrid(x_axis, y_axis, indexing="ij"))
        grid_2d = grid.reshape(2, -1).T

        n_plots = n_samples
        n_rows = (n_plots - 1) // figs_per_row + 1
        n_cols = min(n_plots, figs_per_row)
        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=figsize, constrained_layout=True, sharex=True, sharey=True
        )
        flat_axes = ax.flatten()

        y_samples_flat = self.sample_prior_gp(grid_2d, n_samples=n_samples, random_state=random_state).T

        for i in range(n_samples):
            y_sample_2d = y_samples_flat[i].reshape(grid.shape[1:])

            axx = flat_axes[i]

            im = axx.imshow(
                y_sample_2d,
                origin="lower",
                extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
                cmap=cmap,
            )
            if include_colorbar:
                fig.colorbar(im, ax=axx)

            axx.set_xlabel("X".format(self.units.name))
            axx.set_ylabel("Y".format(self.units.name))
            axx.set_title(f"Prior Sample {i+1}")

        flat_axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self.units.value:.1f}"))
        flat_axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self.units.value:.1f}"))

        for i in range(n_samples, len(flat_axes)):
            flat_axes[i].axis("off")
            
        if artistic:
            for axx in flat_axes:
                axx.axis("off")
                axx.set_title("")
                # remove colorbar
                axx.get_images()[0].colorbar.remove()
                # set the figure background color to black
                fig.patch.set_facecolor('black')
                # set all the text color to white
        return fig, ax

    def plot_kernel_functions(self, kernels: List[Kernel], X: np.ndarray, ax: plt.Axes = None):
        """
        Plot kernel functions.
        :param kernels: List of kernel functions to plot.
        :param X: 2D array of spatial coordinates (shape [n_samples, 2]).
        :param ax: Matplotlib axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        for kernel in kernels:
            K = kernel(X)
            ax.plot(X[:, 0], K[:, 0], label=str(kernel))

        ax.set_xlabel("X")
        ax.set_ylabel("Kernel")
        ax.legend()

    def plot_posterior_gp(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, ax: plt.Axes = None
    ):
        """
        Plot the posterior GP.
        :param X_train: 2D array of training spatial coordinates (shape [n_train, 2]).
        :param y_train: Training target values (shape [n_train]).
        :param X_test: 2D array of test spatial coordinates (shape [n_test, 2]).
        :param ax: Matplotlib axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        self.fit(X_train, y_train)

        y_pred, y_std = self.predict(X_test, return_std=True)

        ax.scatter(X_train[:, 0], y_train, color="red", label="Training data")
        ax.plot(X_test[:, 0], y_pred, color="black", label="Posterior mean")
        ax.fill_between(
            X_test[:, 0],
            y_pred - y_std,
            y_pred + y_std,
            color="gray",
            alpha=0.2,
            label="Posterior std",
        )
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/self.units.value:.1f}"))
        ax.set_xlabel("X [{}]".format(self.units.name))
        ax.set_ylabel("y")
        ax.legend()
