import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import kv, gamma


class SignalAutocorrelation1D:
    def __init__(self, x_values, y_values, normalize=True):
        self.x_values = x_values
        self.y_values = y_values
        self.autocorr = None
        self.normalize = normalize
        self.step = None
        self.steps = None

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

    def calculate_autocorrelation(self):
        y_normalized = self.y_values / np.max(self.y_values) if self.normalize else self.y_values
        self.autocorr = np.correlate(y_normalized, y_normalized, mode="full")[
            len(y_normalized) - 1 :
        ]
        self.step = self.x_values[1] - self.x_values[0]
        self.steps = np.arange(0, len(self.autocorr)) * self.step
        return self.autocorr

    def plot_autocorrelation(self, ax=None, figsize=(10, 3)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        if self.autocorr is None or self.steps is None:
            self.calculate_autocorrelation()
        ax.plot(self.steps, self.autocorr)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.0f}"))
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True)
        return fig, ax
