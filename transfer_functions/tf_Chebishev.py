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
from ..curve.Chebishev1d import ChebyshevFitter1D, Normalization_Info

from my_packages.constants import VoltageUnits, CurrentUnits, DistanceUnits, FrequencyUnits

@dataclass
class _Interpolator:
    coeffs: np.ndarray
    norm_info: Normalization_Info
    chebishev_fitter: ChebyshevFitter1D

    @property
    def f_min(self):
        return self.chebishev_fitter.x_min

    @property
    def f_max(self):
        return self.chebishev_fitter.x_max

    def _assert_if_frequency(self, f: np.ndarray):
        if isinstance(f, float):
            f = np.array([f])
        if f.min() < self.f_min:
            raise ValueError("Frequency below interpolation range")
        if f.max() > self.f_max:
            raise ValueError("Frequency above interpolation range")

    def predict(self, f: float):
        self._assert_if_frequency(f)
        return self.chebishev_fitter.predict(f, self.coeffs, self.norm_info)


class TF:
    def __init__(
        self,
        freq: np.ndarray,
        tf: np.ndarray,
        transfer_function_units: TFUnits = None,
        frequency_units: Union[str, FrequencyUnits] = FrequencyUnits.Hz,
        fitting_order: int = 10,
        fmin: float = None,
        fmax: float = None,
        logx_while_fitting: bool = False,
        logy_while_fitting: bool = False,
    ):
        self.freq = freq * frequency_units.value
        self.tf = tf
        self.frequency_units = frequency_units
        if transfer_function_units is None:
            transfer_function_units = TFUnits(VoltageUnits.V, VoltageUnits.V)
        self.transfer_function_units = transfer_function_units
        self.interpolator = None

        self.logx_while_fitting = logx_while_fitting
        self.logy_while_fitting = logy_while_fitting

        self.fit_function(degree=fitting_order, fmin=fmin, fmax=fmax)

    @property
    def fmin(self):
        return self._fmin

    @property
    def fmax(self):
        return self._fmax

    def fit_function(self, degree: int = 10, fmin: float = None, fmax: float = None):
        if fmin is None:
            fmin = self.freq.min()
        if fmax is None:
            fmax = self.freq.max()

        f = np.log10(self.freq) if self.logx_while_fitting else self.freq
        v = np.log10(self.tf) if self.logy_while_fitting else self.tf

        # set the valid frequency range
        self._fmin = fmin
        self._fmax = fmax

        # if logx_while_fitting update the valid frequency range
        if self.logx_while_fitting:
            fmin = np.log10(fmin)
            fmax = np.log10(fmax)

        cheb_fitter = ChebyshevFitter1D(bounds=(fmin, fmax))

        cheb_fitter.load_data(f, v)
        coeff, norm_info = cheb_fitter.fit(degree)

        self.interpolator = _Interpolator(
            coeffs=coeff,
            norm_info=norm_info,
            chebishev_fitter=cheb_fitter,
        )

    def predict(self, f: np.ndarray):
        if self.logx_while_fitting:
            f = np.log10(f)
        pred = self.interpolator.predict(f)

        if self.logy_while_fitting:
            pred = 10**pred
        return pred

    def inspect(
        self, ax: plt.Axes = None, frequency_units: Union[str, FrequencyUnits] = FrequencyUnits.Hz
    ):
        if isinstance(frequency_units, str):
            frequency_units = FrequencyUnits[frequency_units]

        if ax is None:
            fig = plt.figure(figsize=(10, 4), constrained_layout=True)
            ax = fig.add_subplot()
        else:
            fig = ax.get_figure()

        valid_freq = np.linspace(self.fmin, self.fmax, 1000)

        ax.scatter(self.freq, self.tf, label="original data", c="r", s=25)
        ax.plot(
            valid_freq, self.predict(valid_freq), label="fitting", c="b", linewidth=1
        )

        formatter = FuncFormatter(lambda x, _: "{:.2f}".format(x / frequency_units.value))
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel("Frequency [{}]".format(frequency_units.name))
        ax.set_ylabel("TF dB{}".format(self.transfer_function_units.return_string()))
        # place legend on the lower right corner
        ax.legend(loc="lower right")
        ax.grid(True, which="both")
        ax.set_title("Chebishev Fitting of TF")
        return fig, ax
        # plt.show(block=True)

    @classmethod
    def from_csv(
        cls,
        path: str,
        frequency_units: Union[str, FrequencyUnits],
        transfer_function_units: TFUnits = None,
        fitting_order: int = 10,
        f_min: float = None,
        f_max: float = None,
        logx_while_fitting: bool = False,
        logy_while_fitting: bool = False,
    ) -> "TF":
        df = pd.read_csv(path, names=["f", "v"])
        df["f"] *= frequency_units.value
        df.set_index("f", inplace=True)
        return cls(
            freq=df.index.values,
            tf=df["v"].values,
            frequency_units=FrequencyUnits.Hz,
            transfer_function_units=transfer_function_units,
            fitting_order=fitting_order,
            fmin=f_min,
            fmax=f_max,
            logx_while_fitting=logx_while_fitting,
            logy_while_fitting=logy_while_fitting,
        )
        
    @classmethod
    def from_pd_Series(
        cls, 
        df: pd.Series,
        frequency_units: Union[str, FrequencyUnits],
        transfer_function_units: TFUnits = None,
        fitting_order: int = 10,
        f_min: float = None,
        f_max: float = None,
        logx_while_fitting: bool = False,
        logy_while_fitting: bool = False,
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
            fitting_order=fitting_order,
            fmin=f_min,
            fmax=f_max,
            logx_while_fitting=logx_while_fitting,
            logy_while_fitting=logy_while_fitting,
        )
