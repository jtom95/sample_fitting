import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import kv, gamma
from sklearn.gaussian_process.kernels import Kernel, RBF, Matern, ConstantKernel as C


class ModelKernel1D:
    def __init__(self, x_values, y_values, normalize=True):
        self.x_values = x_values
        self.y_values = y_values
        self.normalize = normalize
        self.constant = None
        self.scale = None
        self.nu = None
        self.kernel = None

    def plot_data(self, ax=None, figsize=(10, 3)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        ax.plot(self.x_values, self.y_values)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        return fig, ax

    def fit_rbf(self, method="Nelder-Mead", bounds=None):
        self.constant, self.scale = self.fit_rbf_length_scale_static(
            self.x_values, self.y_values, method=method, bounds=bounds
        )
        self.kernel = self.constant * RBF(length_scale=self.scale, length_scale_bounds="fixed")
        return self.constant, self.scale

    def fit_multi_rbf(self, n, method="Nelder-Mead", bounds=None):
        self.constant, self.scales, self.weights = self.fit_multi_rbf_parameters_static(
            self.x_values, self.y_values, n, method=method, bounds=bounds
        )
        rbf_list = []
        self.weights = list(self.weights) + [1 - sum(self.weights)]
        for scale, weight in zip(self.scales, self.weights):
            rbf_list.append(weight * RBF(length_scale=scale, length_scale_bounds="fixed"))
        
        self.kernel = self.constant * np.sum(rbf_list)
        return self.constant, self.scales, self.weights

    def fit_matern(self, method="Nelder-Mead", bounds=-1):
        self.constant, self.scale, self.nu = self.fit_matern_parameters_static(
            self.x_values, self.y_values, method=method, bounds=bounds
        )
        self.kernel = self.constant*Matern(
            length_scale=self.scale, length_scale_bounds="fixed", nu=self.nu
        )
        return self.constant, self.scale, self.nu

    def fit_hybrid(self, method="Nelder-Mead", bounds=-1):
        self.constant, self.scale_matern, self.scale_rbf, self.nu, self.r0, self.sigma = (
            self.fit_hybrid_parameters_static(
                self.x_values, self.y_values, method=method, bounds=bounds
            )
        )
        weight1 = 1 / (1 + np.exp((self.x_values[-1] - self.r0) / self.sigma))
        weight2 = 1 - weight1
        self.kernel = self.constant * (
            weight1 * Matern(
                length_scale=self.scale_matern, length_scale_bounds="fixed", nu=self.nu
            )
            + weight2 * RBF(
                length_scale=self.scale_rbf, length_scale_bounds="fixed"
                )
            )

        return self.constant, self.scale_matern, self.scale_rbf, self.nu, self.r0, self.sigma

    def plot_rbf_fit(
        self,
        constant=None,
        scale=None,
        ax=None,
        figsize=(10, 3),
        plot_data=True,
        text_position=(0.8, 0.5),
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        if constant is None:
            if self.constant is None:
                constant = self.y_values[0]
            else:
                constant = self.constant
        if scale is None:
            if self.scale is None:
                raise ValueError("Scale not defined")
            else:
                scale = self.scale
        self.plot_parameters_rbf_static(
            self.x_values,
            self.y_values,
            constant,
            scale,
            ax=ax,
            plot_data=plot_data,
            text_position=text_position,
        )
        return fig, ax

    def plot_multi_rbf_fit(
        self,
        constant=None,
        scales=None,
        weights=None,
        ax=None,
        figsize=(10, 3),
        plot_data=True,
        text_position=(0.8, 0.5),
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        if constant is None:
            if self.constant is None:
                constant = self.y_values[0]
            else:
                constant = self.constant
        if scales is None:
            if self.scales is None:
                raise ValueError("scales not defined")
            else:
                scales = self.scales
        if weights is None:
            if self.weights is None:
                raise ValueError("weights not defined")
            else:
                weights = self.weights

        self.plot_parameters_multi_rbf_static(
            self.x_values,
            self.y_values,
            constant,
            scales,
            weights,
            ax=ax,
            plot_data=plot_data,
            text_position=text_position,
        )
        return fig, ax

    def plot_matern_fit(
        self,
        constant=None,
        scale=None,
        nu=None,
        ax=None,
        figsize=(10, 3),
        plot_data=True,
        text_position=(0.8, 0.5),
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        if constant is None:
            if self.constant is None:
                constant = self.y_values[0]
            else:
                constant = self.constant
        if scale is None:
            if self.scale is None:
                raise ValueError("Scale not defined")
            else:
                scale = self.scale
        if nu is None:
            if self.nu is None:
                raise ValueError("Nu not defined")
            else:
                nu = self.nu
        self.plot_parameters_matern_static(
            self.x_values,
            self.y_values,
            constant,
            scale,
            nu,
            ax=ax,
            plot_data=plot_data,
            text_position=text_position,
        )
        return fig, ax

    def plot_hybrid_fit(
        self,
        constant=None,
        scale_matern=None,
        scale_rbf=None,
        nu=None,
        r0=None,
        sigma=None,
        ax=None,
        figsize=(10, 3),
        plot_data=True,
        text_position=(0.8, 0.4),
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        if constant is None:
            if self.constant is None:
                constant = self.y_values[0]
            else:
                constant = self.constant
        if scale_matern is None:
            if self.scale_matern is None:
                raise ValueError("scale_matern not defined")
            else:
                scale_matern = self.scale_matern
        if scale_rbf is None:
            if self.scale_rbf is None:
                raise ValueError("scale_rbf not defined")
            else:
                scale_rbf = self.scale_rbf
        if nu is None:
            if self.nu is None:
                raise ValueError("nu not defined")
            else:
                nu = self.nu
        if r0 is None:
            if self.r0 is None:
                raise ValueError("r0 not defined")
            else:
                r0 = self.r0
        if sigma is None:
            if self.sigma is None:
                raise ValueError("sigma not defined")
            else:
                sigma = self.sigma

        self.plot_parameters_hybrid_static(
            self.x_values,
            self.y_values,
            constant,
            scale_matern,
            scale_rbf,
            nu,
            r0,
            sigma,
            ax=ax,
            plot_data=plot_data,
            text_position=text_position,
        )
        return fig, ax

    @staticmethod
    def rbf_function(x, constant, scale):
        return constant * np.exp(-(x**2) / (2 * scale**2))

    @staticmethod
    def multi_rbf_kernel(r, constant, scales, weights):
        if len(scales) == len(weights):
            pass
        elif len(scales) == len(weights) + 1:
            weights = list(weights) + [1 - sum(weights)]
        else:
            raise ValueError("The weights must be the same as the number of RBF components minus 1")

        return sum(
            weight * ModelKernel1D.rbf_function(r, constant, scale)
            for scale, weight in zip(scales, weights)
        )

    @staticmethod
    def matern_function(x, constant, scale, nu):
        factor = (2 ** (1 - nu)) / gamma(nu)
        term = (np.sqrt(2 * nu) * x) / scale
        return constant * factor * (term**nu) * kv(nu, term)

    @staticmethod
    def hybrid_kernel(r, constant, nu, scale_matern, scale_rbf, r0, sigma):
        w = 1 / (1 + np.exp((r - r0) / sigma))
        matern = ModelKernel1D.matern_function(r, constant, scale_matern, nu)
        rbf = ModelKernel1D.rbf_function(r, constant, scale_rbf)
        return w * matern + (1 - w) * rbf

    @classmethod
    def fit_multi_rbf_parameters_static(
        cls, x_values, y_values, n, method="Nelder-Mead", bounds=None
    ):
        initial_guess = [np.ptp(x_values) / (3 * (i + 1)) for i in range(n)] + [1 / n] * (
            n - 1
        )  # there are n-1 weights are initialized to 1/n

        constant = y_values[0]

        def objective(params):
            scales = params[:n]
            weights = params[n:]
            predictions = [cls.multi_rbf_kernel(xi, constant, scales, weights) for xi in x_values]
            return sum((yi - pi) ** 2 for yi, pi in zip(y_values, predictions))

        if bounds is None:
            bounds = [(0.5e-3, 10e-2) for _ in range(n)] + [(-1, 1) for _ in range(n - 1)]

        result = minimize(objective, initial_guess, method=method, bounds=bounds)
        return constant, result.x[0:n], result.x[n:]

    @classmethod
    def fit_rbf_length_scale_static(cls, x_values, y_values, method="Nelder-Mead", bounds=None):
        initial_guess = np.ptp(x_values) / 3
        constant = y_values[0]

        def objective(scale):
            predictions = [cls.rbf_function(xi, constant, scale) for xi in x_values]
            return sum((yi - pi) ** 2 for yi, pi in zip(y_values, predictions))

        result = minimize(objective, initial_guess, method=method, bounds=bounds)
        return constant, result.x[0]

    @classmethod
    def fit_matern_parameters_static(
        cls, x_values, y_values, method="Nelder-Mead", bounds=-1, initial_guess=None
    ):
        if initial_guess is None:
            initial_guess = [np.ptp(x_values) / 3, 1.5]
        constant = y_values[0]

        def objective(params):
            scale, nu = params
            predictions = [
                cls.matern_function(xi, constant, scale, nu) for xi in x_values if xi != 0
            ]
            error = sum((yi - pi) ** 2 for yi, pi in zip(y_values, predictions))
            return error

        if bounds == -1:
            bounds = [(0.5e-3, 10e-2), (1, 20)]
        result = minimize(objective, initial_guess, method=method, bounds=bounds)
        return constant, result.x[0], result.x[1]

    @classmethod
    def fit_hybrid_parameters_static(cls, x_values, y_values, method="Nelder-Mead", bounds=-1):
        initial_guess = [np.ptp(x_values) / 3, np.ptp(x_values) / 3, 1.5, np.ptp(x_values) / 2, 1.0]
        constant = y_values[0]

        def objective(params):
            scale_matern, scale_rbf, nu, r0, sigma = params
            predictions = [
                cls.hybrid_kernel(
                    r=xi,
                    constant=constant,
                    nu=nu,
                    scale_matern=scale_matern,
                    scale_rbf=scale_rbf,
                    r0=r0,
                    sigma=sigma,
                )
                for xi in x_values
                if xi != 0
            ]
            error = sum((yi - pi) ** 2 for yi, pi in zip(y_values, predictions))
            return error

        if bounds == -1:
            step = x_values[1] - x_values[0]
            scale_bounds = (step / 2, np.ptp(x_values))
            bounds = [scale_bounds, scale_bounds, (1, 20), scale_bounds, (0.1, 10)]
        result = minimize(objective, initial_guess, method=method, bounds=bounds)
        return constant, result.x[0], result.x[1], result.x[2], result.x[3], result.x[4]

    @classmethod
    def plot_parameters_rbf_static(
        cls, x_values, y_values, constant, scale, ax=None, plot_data=True, text_position=(0.8, 0.5)
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        else:
            fig = ax.get_figure()
        if plot_data:
            ax.plot(x_values, y_values, label="Data")
        ax.plot(
            x_values,
            [cls.rbf_function(xi, constant, scale) for xi in x_values],
            label="RBF fit",
        )
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.text(*text_position, f"Scale: {scale*1e3:.2f} mm", transform=ax.transAxes)
        return fig, ax

    @classmethod
    def plot_parameters_multi_rbf_static(
        cls,
        x_values,
        y_values,
        constant,
        scales,
        weights,
        ax=None,
        plot_data=True,
        text_position=(0.8, 0.5),
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        else:
            fig = ax.get_figure()
        if plot_data:
            ax.plot(x_values, y_values, label="Data")
        ax.plot(
            x_values,
            [cls.multi_rbf_kernel(xi, constant, scales, weights) for xi in x_values],
            label="Multi-RBF fit",
        )
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        scales_text = "\n".join(
            [f"scale_{i+1}: {scale*1e3:.2f} mm" for i, scale in enumerate(scales)]
        )
        weights_text = "\n".join(
            [f"weight_{i+1}: {weight:.2f}" for i, weight in enumerate(weights)]
        )
        ax.text(
            *text_position,
            f"constant: {constant:.2f}\n{scales_text}\n{weights_text}",
            transform=ax.transAxes,
        )
        return fig, ax

    @classmethod
    def plot_parameters_matern_static(
        cls,
        x_values,
        y_values,
        constant,
        scale,
        nu,
        ax=None,
        plot_data=True,
        text_position=(0.8, 0.5),
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        else:
            fig = ax.get_figure()
        if plot_data:
            ax.plot(x_values, y_values, label="Data")
        ax.plot(
            x_values,
            [cls.matern_function(xi, constant, scale, nu) for xi in x_values],
            label="Matern fit",
        )
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.text(*text_position, f"Scale: {scale*1e3:.2f} mm\nNu: {nu:.2f}", transform=ax.transAxes)
        return fig, ax

    @classmethod
    def plot_parameters_hybrid_static(
        cls,
        x_values,
        y_values,
        constant,
        scale_matern,
        scale_rbf,
        nu,
        r0,
        sigma,
        ax=None,
        plot_data=True,
        text_position=(0.8, 0.5),
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        else:
            fig = ax.get_figure()
        if plot_data:
            ax.plot(x_values, y_values, label="Data")
        ax.plot(
            x_values,
            [
                cls.hybrid_kernel(xi, constant, nu, scale_matern, scale_rbf, r0, sigma)
                for xi in x_values
            ],
            label="Hybrid fit",
        )
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.text(
            *text_position,
            f"constant: {constant:.2f}\nscale_matern: {scale_matern*1e3:.2f} mm\nscale_rbf: {scale_rbf*1e3:.2f} mm\nnu: {nu:.2f}\nr0: {r0*1e3:.2f} mm\nsigma: {sigma:.2f}",
            transform=ax.transAxes,
        )
        return fig, ax
