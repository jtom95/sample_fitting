from typing import List, Tuple, Dict, Any, Optional, Union
import os, sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from enum import Enum
from .transfer_functions_aux_classes import TFUnits

# sample fitting imports
from ..curve.Chebishev1d import ChebyshevFitter1D
from ..curve.univariate_fitting import UnivariateFitting
from ..curve.interpolation_fitting import InterpolationFitting
from ..curve.abstract_curve_fitter import AbstractCurveFitter

from my_packages.constants import VoltageUnits, CurrentUnits, DistanceUnits, FrequencyUnits


@dataclass
class FreqModel:
    fitter: AbstractCurveFitter
    check_frequency: bool = True

    @property
    def f_min(self):
        return self.fitter.x_min

    @property
    def f_max(self):
        return self.fitter.x_max

    def _assert_if_frequency(self, f: np.ndarray):
        if isinstance(f, float):
            f = np.array([f])
        if f.min() < self.f_min:
            raise ValueError("Frequency below interpolation range")
        if f.max() > self.f_max:
            raise ValueError("Frequency above interpolation range")

    def predict(self, f: float):
        if self.check_frequency:
            self._assert_if_frequency(f)
        return self.fitter.predict(f)


class TF:
    def __init__(
        self,
        freq: np.ndarray,
        tf: np.ndarray,
        transfer_function_units: TFUnits = None,
        frequency_units: Union[str, FrequencyUnits] = FrequencyUnits.Hz,
        fmin: float = None,
        fmax: float = None,
        logx_while_fitting: bool = False,
        logy_while_fitting: bool = False,
        method: str = "interpolation",
        smoothing: float = None,
        interpolation_kind: str = "cubic",
        order_Cheb: int = 10,
        order_Univariate: int = 3,
        check_freq_bounds: bool = False,
        **kwargs,
    ):
        self.freq = freq * frequency_units.value
        self.tf = tf
        self.frequency_units = frequency_units
        if transfer_function_units is None:
            transfer_function_units = TFUnits(VoltageUnits.V, VoltageUnits.V)
        self.transfer_function_units = transfer_function_units
        self.interpolator: FreqModel = None

        self.logx_while_fitting = logx_while_fitting
        self.logy_while_fitting = logy_while_fitting

        self.check_freq_bounds = check_freq_bounds
        self.kwargs = kwargs

        self.method = method
        if method == "chebyshev":
            self.chebishev_fit_function(degree=order_Cheb, fmin=fmin, fmax=fmax, **kwargs)
        elif method == "univariate":
            self.univariate_fit_function(
                degree=order_Univariate, smoothing_factor=smoothing, **kwargs
            )
        elif method == "interpolation":
            self.interpolation_fitting(kind=interpolation_kind, **kwargs)
        else:
            raise ValueError(
                "method {} not supported. Must be one of 'chebyshev', 'univariate', 'interpolation'".format(
                    method
                )
            )

    @property
    def fmin(self):
        if self.interpolator is None:
            return None
        if self.logx_while_fitting:
            return 10**self.interpolator.f_min

        return self.interpolator.f_min

    @property
    def fmax(self):
        if self.interpolator is None:
            return None
        if self.logx_while_fitting:
            return 10**self.interpolator.f_max
        return self.interpolator.f_max

    def univariate_fit_function(self, degree: int = 3, smoothing_factor: float = None, **kwargs):
        f = np.log10(self.freq) if self.logx_while_fitting else self.freq
        v = np.log10(self.tf) if self.logy_while_fitting else self.tf

        fitter = UnivariateFitting(spline_order=degree, smoothing_factor=smoothing_factor, **kwargs)
        fitter.fit(x=f, y=v)
        self.interpolator = FreqModel(fitter=fitter, check_frequency=self.check_freq_bounds)
        self.fitting_method = "univariate"

    def chebishev_fit_function(
        self, degree: int = 10, fmin: float = None, fmax: float = None, **kwargs
    ):
        if fmin is None:
            fmin = self.freq.min()
        if fmax is None:
            fmax = self.freq.max()

        f = np.log10(self.freq) if self.logx_while_fitting else self.freq
        v = np.log10(self.tf) if self.logy_while_fitting else self.tf

        # # set the valid frequency range
        # self._fmin = fmin
        # self._fmax = fmax

        # if logx_while_fitting update the valid frequency range
        if self.logx_while_fitting:
            fmin = np.log10(fmin)
            fmax = np.log10(fmax)

        cheb_fitter = ChebyshevFitter1D(bounds=(fmin, fmax), **kwargs)
        cheb_fitter.load_data(f, v)
        cheb_fitter.fit(degree)

        self.interpolator = FreqModel(
            fitter=cheb_fitter,
            check_frequency=self.check_freq_bounds,
        )
        self.fitting_method = "chebyshev"

    def interpolation_fitting(self, kind: str = "cubic", **kwargs):
        f = np.log10(self.freq) if self.logx_while_fitting else self.freq
        v = np.log10(self.tf) if self.logy_while_fitting else self.tf

        fitter = InterpolationFitting(kind=kind, **kwargs)
        fitter.fit(x=f, y=v)
        self.interpolator = FreqModel(fitter=fitter, check_frequency=self.check_freq_bounds)
        self.fitting_method = "interpolation"

    def predict(self, f: np.ndarray):
        if self.logx_while_fitting:
            f = np.log10(f)
        pred = self.interpolator.predict(f)

        if self.logy_while_fitting:
            pred = 10**pred
        return pred

    def inspect(
        self,
        ax: plt.Axes = None,
        frequency_units: Union[str, FrequencyUnits] = FrequencyUnits.Hz,
        fbounds=None,
    ):
        if isinstance(frequency_units, str):
            frequency_units = FrequencyUnits[frequency_units]

        if ax is None:
            fig = plt.figure(figsize=(10, 4), constrained_layout=True)
            ax = fig.add_subplot()
        else:
            fig = ax.get_figure()

        if fbounds is None:
            fbounds = (self.fmin, self.fmax)
        valid_freq = np.linspace(*fbounds, 1000)

        # valid_freq = np.linspace(self.fmin, self.fmax, 1000)

        ax.scatter(self.freq, self.tf, label="original data", c="r", s=25)
        ax.plot(valid_freq, self.predict(valid_freq), label="fitting", c="b", linewidth=1)

        formatter = FuncFormatter(lambda x, _: "{:.2f}".format(x / frequency_units.value))
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel("Frequency [{}]".format(frequency_units.name))
        ax.set_ylabel("TF dB{}".format(self.transfer_function_units.return_string()))
        # place legend on the lower right corner
        ax.legend(loc="lower right")
        ax.grid(True, which="both")
        ax.set_title(f"{self.fitting_method.capitalize()} Fitting of TF")
        return fig, ax
        # plt.show(block=True)

    @classmethod
    def from_csv(
        cls,
        path: str,
        frequency_units: Union[str, FrequencyUnits],
        transfer_function_units: TFUnits = None,
        f_min: float = None,
        f_max: float = None,
        logx_while_fitting: bool = False,
        logy_while_fitting: bool = False,
        method: str = "interpolation",
        smoothing: float = None,
        interpolation_kind: str = "cubic",
        order_Cheb: int = 10,
        order_Univariate: int = 3,
        check_freq_bounds: bool = True,
        **kwargs,
    ) -> "TF":
        df = pd.read_csv(path, names=["f", "v"])
        df["f"] *= frequency_units.value
        df.set_index("f", inplace=True)
        return cls(
            freq=df.index.values,
            tf=df["v"].values,
            frequency_units=FrequencyUnits.Hz,
            transfer_function_units=transfer_function_units,
            fmin=f_min,
            fmax=f_max,
            logx_while_fitting=logx_while_fitting,
            logy_while_fitting=logy_while_fitting,
            method=method,
            smoothing=smoothing,
            interpolation_kind=interpolation_kind,
            order_Cheb=order_Cheb,
            order_Univariate=order_Univariate,
            check_freq_bounds=check_freq_bounds,
            **kwargs,
        )

    @classmethod
    def from_pd_Series(
        cls,
        df: pd.Series,
        frequency_units: Union[str, FrequencyUnits],
        transfer_function_units: TFUnits = None,
        f_min: float = None,
        f_max: float = None,
        logx_while_fitting: bool = False,
        logy_while_fitting: bool = False,
        method: str = "interpolation",
        smoothing: float = None,
        interpolation_kind: str = "cubic",
        order_Cheb: int = 10,
        order_Univariate: int = 3,
        check_freq_bounds: bool = False,
        **kwargs,
    ) -> "TF":
        if isinstance(frequency_units, str):
            frequency_units = FrequencyUnits[frequency_units]
        df.index *= frequency_units.value
        # name the column to "v" for consistency with the from_csv method
        # df.columns = ["v"]
        return cls(
            freq=df.index.values.astype(float),
            tf=df.values,
            frequency_units=FrequencyUnits.Hz,
            transfer_function_units=transfer_function_units,
            fmin=f_min,
            fmax=f_max,
            logx_while_fitting=logx_while_fitting,
            logy_while_fitting=logy_while_fitting,
            method=method,
            smoothing=smoothing,
            interpolation_kind=interpolation_kind,
            order_Cheb=order_Cheb,
            order_Univariate=order_Univariate,
            check_freq_bounds=check_freq_bounds,
            **kwargs,
        )
