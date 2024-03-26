from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion


class NoisePowerExtractorPlotMixin:
    def plot_data(
        self,
        scan_idx=0,
        normalized=False,
        ax=None,
        figsize=(8, 7),
        aspect="auto",
        cmap="jet",
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        data = self.normalized_data if normalized else self.data
        q = ax.imshow(data[:, :, scan_idx], aspect=aspect, cmap=cmap, **kwargs)
        fig.colorbar(q, ax=ax)
        ax.set_title(f"Data (Scan {scan_idx})")
        return ax

    def plot_fft(
        self,
        scan_idx=0,
        log_scale=True,
        ax=None,
        figsize=(8, 7),
        aspect="auto",
        cmap="jet",
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        fft_data = np.abs(self.fft_analyzer.fft_data[:, :, scan_idx])
        if log_scale:
            fft_data = np.log(fft_data + 1e-10)
        q = ax.imshow(fft_data, aspect=aspect, cmap=cmap, **kwargs)
        fig.colorbar(q, ax=ax)
        ax.set_title(f"FFT (Scan {scan_idx})")
        return ax

    def plot_envelope(
        self, scan_idx=0, ax=None, figsize=(8, 7), aspect="auto", cmap="jet", **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        envelope = self.make_envelope()
        q = ax.imshow(envelope[:, :, scan_idx], aspect=aspect, cmap=cmap, **kwargs)
        fig.colorbar(q, ax=ax)
        ax.set_title(f"Envelope (Scan {scan_idx})")
        return ax

    def plot_cutoff_over_fft(
        self, scan_idx=0, ax=None, figsize=(8, 7), aspect="auto", cmaps: Literal["type1", "type2"] = "type1", **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        envelope = self.make_envelope()
        last_ring_mask = self.find_last_ring_mask(envelope).astype(float) # if not cast to float, the nan values will be converted to 0
        # set the values of last ring mask to nan where the envelope is zero
        last_ring_mask[last_ring_mask == 0] = np.nan

        if cmaps == "type1":
            cmap = "jet"
            cmap_cutoff = "binary"
        elif cmaps == "type2":
            cmap = "viridis"
            cmap_cutoff = "binary"

        fft_data = np.abs(self.fft_analyzer.fft_data[:, :, scan_idx])
        q = ax.imshow(fft_data, aspect=aspect, cmap=cmap, **kwargs)
        q_cutoff = ax.imshow(last_ring_mask[:, :, scan_idx], aspect=aspect, cmap=cmap_cutoff, **kwargs)
        fig.colorbar(q, ax=ax)
        ax.set_title(f"Cutoff Over FFT (Scan {scan_idx})")
        return ax

    def plot_low_freq(
        self, scan_idx=0, ax=None, figsize=(8, 7), aspect="auto", cmap="jet", **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        low_freq, _ = self.separate_frequencies(normalize=False)
        q = ax.imshow(np.abs(low_freq[:, :, scan_idx]), aspect=aspect, cmap=cmap, **kwargs)
        fig.colorbar(q, ax=ax)
        ax.set_title(f"Low Frequency (Scan {scan_idx})")
        return ax

    def plot_high_freq(
        self, scan_idx=0, ax=None, figsize=(8, 7), aspect="auto", cmap="jet", **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        _, high_freq = self.separate_frequencies(normalize=False)
        q = ax.imshow(np.abs(high_freq[:, :, scan_idx]), aspect=aspect, cmap=cmap, **kwargs)
        fig.colorbar(q, ax=ax)
        ax.set_title(f"High Frequency (Scan {scan_idx})")
        return ax

    def plot_ifft(
        self,
        scan_idx=0,
        with_margins=True,
        ax=None,
        figsize=(8, 7),
        aspect="auto",
        cmap="jet",
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        _, high_freq = self.separate_frequencies(normalize=False)
        high_freq = high_freq - np.mean(high_freq, axis=(0, 1))
        reconstructed = self.fft_analyzer.ifft_transform(high_freq)

        if not with_margins:
            marginx = np.round(self.margin_ratio * reconstructed.shape[0]).astype(int)
            marginy = np.round(self.margin_ratio * reconstructed.shape[1]).astype(int)
            reconstructed = reconstructed[marginx:-marginx, marginy:-marginy, :]

        # make sure the y axis starts from the bottom
        q = ax.imshow(reconstructed[:, :, scan_idx], aspect=aspect, cmap=cmap, origin='lower', **kwargs)
        fig.colorbar(q, ax=ax)
        ax.set_title(f"IFFT (Scan {scan_idx}, {'With' if with_margins else 'Without'} Margins)")
        return ax

    def plot_noise_powers(
        self, method: Literal["fft", "ifft"] = "ifft", 
        ax=None, figsize=(6, 3), 
        **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        noise_powers = self.estimate_noise_powers(normalize=True, method=method)
        ax.plot(noise_powers, **kwargs)
        ax.set_xlabel("Scan Index")
        ax.set_ylabel("Noise Power")
        ax.set_title("Estimated Noise Powers")
        return ax

    def plot_cutoff_harmonic(
        self, scan_idx=0, ax=None, figsize=(8, 7), aspect="auto", cmap="jet", **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

        else:
            fig = ax.get_figure()

        last_ring_mask = self.find_last_ring_mask()
        # Create a spatial domain representation of the last ring harmonic
        harmonic_spatial = self.fft_analyzer.ifft_transform(last_ring_mask)[..., scan_idx]

        # normalize the harmonic spatial

        # remove the margins
        marginx = np.round(self.margin_ratio * harmonic_spatial.shape[0]).astype(int)
        marginy = np.round(self.margin_ratio * harmonic_spatial.shape[1]).astype(int)
        harmonic_spatial = harmonic_spatial[marginx:-marginx, marginy:-marginy]

        harmonic_spatial = (harmonic_spatial - np.min(harmonic_spatial)) / (np.max(harmonic_spatial) - np.min(harmonic_spatial))

        q = ax.imshow(harmonic_spatial, aspect=aspect, cmap=cmap, **kwargs)
        fig.colorbar(q, ax=ax)
        ax.set_title(f"Cutoff Harmonic (Spatial Domain) (Scan {scan_idx})")

        return ax

    def plot_last_ring_harmonic_fft(
        self, scan_idx=0, ax=None, figsize=(8, 7), aspect="auto", cmap="binary_r", **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()

        last_ring_mask = self.find_last_ring_mask()

        q = ax.imshow(last_ring_mask[..., scan_idx], aspect=aspect, cmap=cmap, **kwargs)
        fig.colorbar(q, ax=ax)
        ax.set_title(f"Last Ring Harmonic (Spatial Domain) (Scan {scan_idx})")
        return ax

    def find_last_ring_mask(self, envelope=None):
        if envelope is None:
            envelope = self.make_envelope()
        # Find the boundary between ones and zeros in the envelope
        last_ring_mask = np.zeros_like(envelope)
        for ff in range(envelope.shape[-1] - 1):
            eroded_envelope = binary_erosion(envelope[..., ff])
            last_ring_mask[..., ff] = np.logical_xor(envelope[..., ff], eroded_envelope)
        return last_ring_mask

    def plot_signal_powers(
        self, method: Literal["fft", "ifft"] = "ifft", ax=None, figsize=(6, 3), **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        signal_powers = self.estimate_signal_powers(normalize=True, method=method)
        ax.plot(signal_powers, **kwargs)
        ax.set_xlabel("Scan Index")
        ax.set_ylabel("Signal Power")
        ax.set_title("Estimated Signal Powers")
        return ax

    def plot_snr(self, dB: bool=True, method: Literal["fft", "ifft"] ="ifft", ax=None, figsize=(6, 3), **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()
        snr = self.estimate_snr(normalize=True, method=method, dB=dB)
        ax.plot(snr, **kwargs)
        ax.set_xlabel("Scan Index")
        if dB:
            ax.set_ylabel("SNR (dB)")
        else:
            ax.set_ylabel("SNR")
        ax.set_title("Estimated SNR")
        return ax
