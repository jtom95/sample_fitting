import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy
from typing import Literal, Optional
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Kernel

from my_packages.high_level_apps.dipole_field_calculator import HighLevelDipoleFieldCalculator
from my_packages.classes.dipole_fields import Substrate
from ..kernel_fit.signals1d import ModelKernel1D
from ...correlation.autocorrelation1d import SignalAutocorrelation1D


class DipoleFieldKernelExtractor:
    MAX_DIM_RANGE = 0.2
    def __init__(
        self,
        f: float,
        dipole_height: float,
        substrate: Substrate,
        dim_range: Optional[float] = None,
        num_points: int = 1000,
        probe_height: float = 1e-2,
    ):
        self.f = f
        self.dipole_height = dipole_height
        self.substrate = substrate
        self.dim_range = dim_range
        self.num_points = num_points
        self.probe_height = probe_height

        self.length_scales = {}
        self.signals = {}
        self.x_values = {}
        self.model_kernel_signals = {}
        self.kernels = {}
        self.kernel_classes = {}

        self._range_in_use = None

    @property
    def range_in_use(self):
        if self.dim_range is not None:
            return self.dim_range
        elif self._range_in_use is not None:
            return self._range_in_use
        else:
            return None

    @property
    def x(self):
        return np.linspace(-self.range_in_use / 2, self.range_in_use / 2, self.num_points)

    @property
    def y(self):
        return np.zeros(self.num_points)

    @property
    def z(self):
        return np.ones(self.num_points) * self.probe_height

    @property
    def r_x(self):
        return np.array((self.x, self.y, self.z)).T

    @property
    def r_y(self):
        return np.array((self.y, self.x, self.z)).T

    @property
    def Ez_kernel(self)->Kernel:
        return self.kernels["Ez"]
    @property
    def Hy_kernel(self)->Kernel:
        kernel_properties = self._get_magnetic_kernel_components()
        return self._make_asynotropic_kernel(**kernel_properties)
    @property
    def Hx_kernel(self)->Kernel:
        kernel_properties = self._get_magnetic_kernel_components()
        weights_x = kernel_properties["weights_y"]
        lengths_x = kernel_properties["lengths_y"]
        weights_y = kernel_properties["weights_x"]
        lengths_y = kernel_properties["lengths_x"]
        return self._make_asynotropic_kernel(lengths_x, weights_x, lengths_y, weights_y)

    def _get_magnetic_kernel_components(self):
        if isinstance(self.length_scales["Hy_x"], dict):
            num_scales = len(self.length_scales["Hy_x"]) // 2
            lengths_x = [self.length_scales["Hy_x"][f"scale_{ii}"] for ii in range(num_scales)]
            weights_x = [self.length_scales["Hy_x"][f"weight_{ii}"] for ii in range(num_scales)]
        else: 
            lengths_x = [self.length_scales["Hy_x"]]
            weights_x = [1]

        if isinstance(self.length_scales["Hy_y"], dict):
            num_scales = len(self.length_scales["Hy_y"]) // 2
            lengths_y = [self.length_scales["Hy_y"][f"scale_{ii}"] for ii in range(num_scales)]
            weights_y = [self.length_scales["Hy_y"][f"weight_{ii}"] for ii in range(num_scales)]
        else:
            lengths_y = [self.length_scales["Hy_y"]]
            weights_y = [1]
        return {
            "lengths_x": lengths_x,
            "weights_x": weights_x,
            "lengths_y": lengths_y,
            "weights_y": weights_y
        }

    def _make_asynotropic_kernel(self, lengths_x, weights_x, lengths_y, weights_y):
        kernels = []
        for (scale_x, weight_x), (scale_y, weight_y) in product(zip(lengths_x, weights_x), zip(lengths_y, weights_y)):
            combined_weight = weight_x * weight_y
            kernels.append(RBF(length_scale=[scale_x, scale_y], length_scale_bounds="fixed") * combined_weight)
        final_kernel = np.sum(kernels)
        return final_kernel

    def _calculate_dipole_main_field(
        self,
        base_dipole: Literal["Ez", "Hx", "Hy"] = "Hy",
        axis: Literal["x", "y"] = "y",
        normalize: bool = True,
        dim_range: float = None,
    ):
        if dim_range is not None:
            x = np.linspace(-dim_range / 2, dim_range / 2, self.num_points)
            y = np.zeros(self.num_points)
            z = np.ones(self.num_points) * self.probe_height
            points = np.array((x, y, z)).T if axis == "x" else np.array((y, x, z)).T
        else:
            if self.dim_range is None: 
                self._range_in_use = self._calculate_default_range(base_dipole=base_dipole, axis=axis)    
            points = self.r_x if axis == "x" else self.r_y
        
        axis_points = points[:, 0] if axis == "x" else points[:, 1]

        calc = HighLevelDipoleFieldCalculator.init_from_single_dipole(
            x=0,
            y=0,
            base_dipole=base_dipole,
            moment=1e-6,
            f=self.f,
            dipole_height=self.dipole_height,
            substrate=self.substrate,
        )

        if base_dipole == "Ez":
            dipole_field = np.abs(calc.evaluate_E(points=points)[-1])
        elif base_dipole == "Hx":
            dipole_field = np.abs(calc.evaluate_H(points=points)[0])
        elif base_dipole == "Hy":
            dipole_field = np.abs(calc.evaluate_H(points=points)[1])
        if normalize:
            return dipole_field / np.max(dipole_field), axis_points 
        return dipole_field, axis_points

    def _calculate_dipole_1d_signal(
        self,
        signal_type: Literal["Ez", "Hy_x", "Hy_y"] = "Ez",
        normalize=True,
        dim_range: float = None,
    ):
        if dim_range is not None:
            range_to_use = dim_range
        elif self.dim_range is None:
            component = signal_type.split("_")[0]
            axis = "x" if signal_type=="Hy_x" else "y"
            range_to_use = self._calculate_default_range(base_dipole=component, axis=axis)
        else:
            range_to_use = self.dim_range

        if signal_type == "Ez":
            return self._calculate_dipole_main_field(
                base_dipole="Ez", axis="x", normalize=normalize, dim_range=range_to_use
            )
        if signal_type == "Hy_x":
            return self._calculate_dipole_main_field(
                base_dipole="Hy", axis="x", normalize=normalize, dim_range=range_to_use
            )
        if signal_type == "Hy_y":
            return self._calculate_dipole_main_field(
                base_dipole="Hy", axis="y", normalize=normalize, dim_range=range_to_use
            )

    def _calculate_default_range(
        self,
        base_dipole: Literal["Ez", "Hx", "Hy"] = "Hy",
        axis: Literal["x", "y"] = "y",
        ratio: float = 1 / 40,
        max_iterations: int = 20,
        tolerance: float = 1e-4,
    ):
        calc = HighLevelDipoleFieldCalculator.init_from_single_dipole(
            x=0,
            y=0,
            base_dipole=base_dipole,
            moment=1e-6,
            f=self.f,
            dipole_height=self.dipole_height,
            substrate=self.substrate,
        )

        def evaluate_field(points):
            if base_dipole == "Ez":
                return np.abs(calc.evaluate_E(points=points)[-1])
            elif base_dipole == "Hx":
                return np.abs(calc.evaluate_H(points=points)[0])
            elif base_dipole == "Hy":
                return np.abs(calc.evaluate_H(points=points)[1])

        max_field = evaluate_field(np.array([[0, 0, self.probe_height]]))[0]
        target_field = max_field * ratio

        left, right = 0, self.MAX_DIM_RANGE
        for _ in range(max_iterations):
            mid = (left + right) / 2
            if axis == "x":
                points = np.array([[-mid, 0, self.probe_height], [mid, 0, self.probe_height]])
            else:
                points = np.array([[0, -mid, self.probe_height], [0, mid, self.probe_height]])

            fields = evaluate_field(points)
            if np.abs(fields[0] - target_field) < tolerance and np.abs(fields[1] - target_field) < tolerance:
                break
            elif fields[0] > target_field:
                left = mid
            else:
                right = mid

        return 2*mid

    def evaluate_single_dipole_signal(
        self,
        signal_type: Literal["Ez", "Hy_x", "Hy_y"] = "Ez",
        dim_range: float = None,
        normalize=True,
    ):
        if signal_type not in self.signals:
            self.signals[signal_type], self.x_values[signal_type] = self._calculate_dipole_1d_signal(
                signal_type=signal_type, normalize=normalize, dim_range=dim_range
            )
        return self

    def evaluate_dipole_signals(self, normalize=True):
        for signal_type in ["Ez", "Hy_x", "Hy_y"]:
            self.evaluate_single_dipole_signal(signal_type=signal_type, normalize=normalize)
        return self

    def calculate_length_scale(
        self,
        field_component: Literal["Ez", "Hy_x", "Hy_y"] = "Ez",
        n_rbfs: int = 1,
    ):
        if field_component not in self.signals:
            self.evaluate_single_dipole_signal(signal_type=field_component)

        signal = self.signals[field_component]
        x_values = self.x_values[field_component]

        acorr = SignalAutocorrelation1D(
            x_values=x_values,
            y_values=signal,
        )
        acorr.calculate_autocorrelation()

        n = len(x_values) // 2 - 1
        model_kernel_signal = ModelKernel1D(
            x_values=acorr.lags[: n + 2],
            y_values=signal[n:],
        )
        if n_rbfs > 1:
            model_kernel_signal.fit_multi_rbf(n=n_rbfs)
            length_scale = {}
            for ii, (scale, weight) in enumerate(zip(model_kernel_signal.scales, model_kernel_signal.weights)):
                length_scale[f"scale_{ii}"] = scale
                length_scale[f"weight_{ii}"] = weight
        else:
            model_kernel_signal.fit_rbf()
            length_scale = deepcopy(model_kernel_signal.scale)

        self.length_scales[field_component] = length_scale
        self.model_kernel_signals[field_component] = model_kernel_signal
        self.kernels[field_component] = deepcopy(model_kernel_signal.kernel)
        self.kernel_classes[field_component] = model_kernel_signal
        return length_scale

    def calculate_length_scales(self):
        for field_component in ["Ez", "Hy_x", "Hy_y"]:
            self.calculate_length_scale(field_component=field_component)
        return self

    def plot_signal(self, signal_type: Literal["Ez", "Hy_x", "Hy_y"], ax=None, figsize=(6, 4)):
        if signal_type not in self.signals:
            self.evaluate_single_dipole_signal(signal_type=signal_type)

        signal = self.signals[signal_type]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()

        ax.plot(self.x * 1e3, signal, label=signal_type)
        ax.set_xlabel("Distance (mm)")
        ax.set_ylabel("Normalized Amplitude")
        ax.legend()
        ax.grid(True)

        return fig, ax

    def plot_all_signals(self, figsize=(10, 6)):
        fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        fig.suptitle("Dipole Field Signals")

        for ax, signal_type in zip(axs, ["Ez", "Hy_x", "Hy_y"]):
            self.plot_signal(signal_type, ax=ax)

        return fig, axs

    def plot_fitting(
        self,
        signal_type: Literal["Ez", "Hy_x", "Hy_y"],
        ax=None,
        figsize=(6, 4),
        text_position=(0.7, 0.5),
    ):
        if signal_type not in self.model_kernel_signals:
            self.calculate_length_scale(field_component=signal_type)

        model_kernel_signal = self.model_kernel_signals[signal_type]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()

        fig.suptitle(f"{signal_type} fitting on single dipole")

        if self.length_scales is None or self.length_scales.get(signal_type) is None:
            raise ValueError("Length scales not calculated")
        if isinstance(self.length_scales[signal_type], dict):
            self.kernel_classes[signal_type].plot_multi_rbf_fit(text_position=text_position, ax=ax)
        else:
            self.kernel_classes[signal_type].plot_rbf_fit(ax=ax, text_position=text_position)

        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.1f}"))

        ax.set_xlabel("lag (mm)")
        ax.set_ylabel("Normalized Amplitude")

        ax.grid(which="both", linestyle="--", linewidth=0.5)

        return fig, ax

    def plot_all_fittings(self, figsize=(15, 5)):
        fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        fig.suptitle("Dipole Field Fittings")

        for ax, signal_type in zip(axs, ["Ez", "Hy_x", "Hy_y"]):
            self.plot_fitting(signal_type, ax=ax)

        return fig, axs
