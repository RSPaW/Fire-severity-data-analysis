#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from scipy.linalg import svd
from scipy.stats import binned_statistic
from scipy import fft
from astropy.timeseries import LombScargle

from numpy.lib.format import open_memmap

"""
With our data set:
- 369 time steps,
- sparse events per month,
- moving gaps (i.e., no-data regions differ but only slightly),
- a very large spatial domain,
an EOF/PCA decomposition lets us identify the dominant spatio-temporal
patterns without having to solve the spectral estimation problem up front
while having to account for missing-data. Challenging because the null 
observations or masking areas tend to dominate. 
"""


def load_severity_matrix(severity_path: str, timestamps_path: str):
    """Load the severity data as a mem-mapped file (i.e., not all in memory)"""
    severity = open_memmap(severity_path, mode="r")
    tstamps = np.load(timestamps_path)

    # tstamps has the shape: (T,) where:
    # - T is the number of time steps in the data

    # severity has the shape: (T, Y, X, M) where:
    # - Y is the number of pixels in the y-dimension
    # - X is the number of pixels in the x-dimension
    # - M is the severity metric (0th index) or the fire-type (1st index)

    assert (
        severity.shape[0] == tstamps.shape[0]
    ), "Time dimension mismatch between severity and timestamps"
    print(f"Severity shape: {severity.shape}")
    print(f"Timestamps shape: {tstamps.shape}")

    return severity, tstamps


def measure_mask_differences(arr, metric=0):
    """Inspect the masking of each frame (time step)"""

    print("Measuring changes in masking w.r.t. first frame")
    mask0 = arr[0, :, :, metric] != -1
    changes = []

    for t in range(arr.shape[0]):
        mask = arr[t, :, :, metric] != -1
        frac_changed = np.mean(mask != mask0)
        changes.append(frac_changed)

    print("Difference in masking fraction:")
    print("  min : median : max ")
    print(
        f"  {np.min(changes):g} : {np.median(changes):g} : {np.max(changes):g}"
    )


def explore_data(arr, times, metric=0):

    nt, ny, nx, _ = arr.shape

    # Plot observation frequency per pixel, and then use this to construct a
    # mask of pixel that are always observed.
    valid_count = np.zeros((ny, nx), dtype=np.uint16)
    burn_count = np.zeros((ny, nx), dtype=np.uint16)
    for t in range(nt):
        valid_count += arr[t, :, :, metric] != -1
        burn_count += arr[t, :, :, metric] > 0

    repeated_burns = burn_count > 3
    frac_observed = valid_count / nt

    # For the next parts, consider only pixels where they are covered for
    # >99% of all time steps.
    mask = frac_observed > 0.99

    # Plot some field observables and general behaviour across the region
    plt.imshow(frac_observed, interpolation="none")
    plt.colorbar(label="Fraction of timesteps observed")
    plt.savefig("severity_frac_observation_map.png", bbox_inches="tight")
    plt.clf()

    plt.imshow(valid_count, interpolation="none")
    plt.colorbar(label="Number of timesteps observed")
    plt.savefig("severity_count_observation_map.png", bbox_inches="tight")
    plt.clf()

    plt.imshow(np.where(mask, burn_count, np.nan), interpolation="none")
    plt.title("Fire event counts with severity >= 1")
    plt.colorbar(label="Number of fire events")
    plt.savefig("severity_count_burns_map.png", bbox_inches="tight")
    plt.clf()

    plt.imshow(repeated_burns, interpolation="none")
    plt.title("Fire repetition count")
    plt.colorbar(label="More than 3 fire events recorded?")
    plt.savefig("severity_repeated_burn_map.png", bbox_inches="tight")
    plt.clf()

    # Compute the field activity (sum of pixel) and the conditional sum to
    # identify often-active places and/or places are are rarely-active but are
    # intense when active.Plot the activity of the field over time
    act = np.zeros(nt)
    cond_sum = np.zeros((ny, nx))
    for t in range(nt):
        frame = arr[t, :, :, metric].copy().astype(float)
        masked_frame = np.where(mask, frame, np.nan)
        act[t] = np.nansum(masked_frame)
        cond_sum = np.nansum(np.dstack((cond_sum, masked_frame)), axis=2)

    plt.vlines(times, ymin=0, ymax=act / 1e5)
    plt.xlabel("Date")
    plt.ylabel("Activity (pixel sum $\times 10^5$)")
    plt.savefig("severity_activity_over_time.png", bbox_inches="tight")
    plt.clf()

    plt.imshow(np.where(mask, cond_sum, np.nan), interpolation="none")
    plt.colorbar(label="Severity sum over time")
    plt.savefig("severity_sum_map.png", bbox_inches="tight")
    plt.clf()


def block_reduce_mean(arr, metric=0, factor=20):
    """Compute the mean over valid pixels only and ignores the -1 sentinel
    locations entirely. This will write 2 new files and memmap them."""

    nt, ny, nx, _ = arr.shape

    # scale to nearest factor
    ny2 = (ny // factor) * factor
    nx2 = (nx // factor) * factor

    # get reduced cell shape
    nyc = ny2 // factor
    nxc = nx2 // factor

    # open the reduced/coarse data file
    Y = np.memmap(
        "coarse_data.dat",
        mode="w+",
        dtype=np.float32,
        shape=(nt, nyc, nxc),
    )

    # open the coverage data file
    W = np.memmap(
        "coarse_coverage.dat",
        mode="w+",
        dtype=np.float32,
        shape=(nt, nyc, nxc),
    )
    # Note, the above will have created two files on disk.
    # Feel free to remove them after execution is desired.
    # They could alternatively fit in RAM (only about 350 MB each), so may not
    # need to mem-map them, in which case just create the shaped numpy arrays.

    for t in range(nt):
        frame = arr[t, :ny2, :nx2, metric]
        valid = frame != -1

        vals = np.where(valid, frame, 0)
        sums = vals.reshape(nyc, factor, nxc, factor).sum(axis=(1, 3))
        counts = valid.reshape(nyc, factor, nxc, factor).sum(axis=(1, 3))

        coarse = np.full((nyc, nxc), np.nan, dtype=np.float32)

        good = counts > 0

        coarse[good] = sums[good] / counts[good]

        Y[t] = coarse
        W[t] = counts / (factor * factor)

    # mean_coverage = np.nanmean(W, axis=0)
    # plt.imshow(mean_coverage, interpolation="none")
    # plt.colorbar(label="Mean coverage")
    # plt.show()
    # plt.clf()

    return Y, W


def remove_mean_and_seasonality(arr: np.ndarray, times: np.ndarray):
    """Remove the mean and seasonality from the data. This is a common
    preprocessing step for time series analysis."""

    # Look for the anomolies, so remove mean
    anom = arr - np.mean(arr, axis=0)

    # Remove seasonality (assuming monthly data)
    # converting numpy datetime64 objects into months:
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    for m in range(1, 13):
        idx = months == m
        clim = np.mean(arr[idx], axis=0)
        anom[idx] -= clim

    return anom


def radial_spectrum(field, dx, nbins=150):
    """Produce a radial spatial spectrum."""
    ny, nx = field.shape

    f = fft.rfft2(np.nan_to_num(field), workers=-1)
    power = np.abs(f) ** 2
    ky = 2 * np.pi * fft.fftfreq(ny, d=dx)
    kx = 2 * np.pi * fft.rfftfreq(nx, d=dx)

    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)

    bins = np.linspace(0, kr.max(), nbins + 1)

    Pk, _, _ = binned_statistic(
        kr.ravel(), power.ravel(), statistic="mean", bins=bins
    )

    k = 0.5 * (bins[:-1] + bins[1:])

    return k, Pk


def make_severity_map_movie(arr):
    import imageio

    vmin = -1
    vmax = 5

    writer = imageio.get_writer("movie.mp4", fps=2, codec="libx264")
    for i in range(arr.shape[0]):
        img = arr[i, ..., 0].copy()
        img8 = (img - vmin) / (vmax - vmin) * 255
        img8 = np.clip(img8, 0, 255).astype(np.uint8)
        writer.append_data(img8)
    writer.close()


if __name__ == "__main__":
    print("Loading severity data and timestamps")
    severity, tstamps = load_severity_matrix("severity.npy", "timestamps.npy")
    nt, ny, nx, nm = severity.shape
    xy_scale = 30  # each pixel is above 30m x 30m

    print("Making a movie from the time steps... (might take a while)")
    # NOTE: This can take a while, so comment out if you're interested in
    # the other stuff!
    # make_severity_map_movie(severity)

    print("Exploring data... (might take a while)")
    # NOTE: This can take a while, so comment out if you're interested in
    # the other stuff!
    # measure_mask_differences(severity)
    # explore_data(severity, tstamps)

    # Reduce the size of the data and compute the anomoly
    reduction_factor = 10
    print(f"Reducing pixel scale by factor={reduction_factor}")
    y, w = block_reduce_mean(severity, factor=reduction_factor)
    coarse_ny, coarse_nx = y[0, ...].shape
    good_cells = np.nanmean(w, axis=0) > 0.8
    print("Computing the anomoly (removing mean seasonality)")
    y_anom = remove_mean_and_seasonality(y, tstamps)

    # Reshape the coarse data in preparation for PCA
    print("Preparing data for PCA...")
    y2 = y_anom.reshape(nt, coarse_ny * coarse_nx)
    good_cells_flat = good_cells.ravel()
    y2 = y2[:, good_cells_flat]

    # Fill in NaNs with the column means, as these will not show-up as a
    # changing component.
    column_mean = np.nanmean(y2, axis=0)
    nan_idx = np.isnan(y2)
    y2[nan_idx] = column_mean[nan_idx[1]]

    # Use SVD to perform the PCA
    print("Using SVD to execute PCA decomposition")
    U, s, VT = svd(y2, full_matrices=False)
    temporal_pc = U * s  # aka "scores"
    spatial_eofs = VT
    varfrac = s**2 / np.sum(s**2)
    loadings = spatial_eofs.T
    print(f"Scores shape: {temporal_pc.shape}")
    print(f"Loadings shape: {loadings.shape}")

    # Compute the radial spatial spectra
    # (i.e., "distance" rather than specific x-y offsets)
    spectra = []
    modes = []
    for t in range(nt):
        k, Pk = radial_spectrum(y_anom[t], xy_scale * reduction_factor)
        spectra.append(Pk)
        modes.append(k)
    spectra = np.asarray(spectra)
    modes = np.asarray(modes)

    print("Generating diagnostic plots...")
    # Look at the total variance explained as a function of component
    print(varfrac.shape)
    varplot = plt.plot(np.cumsum(varfrac), marker="o")
    plt.gca().xaxis.set_major_locator(mt.MaxNLocator(integer=True))
    plt.ylabel("Cumulative variance explained")
    plt.xlabel("PC index")
    plt.savefig("pca_variance_explained.png", bbox_inches="tight")
    plt.clf()

    # Bi-plot(s) to identify how PCAs relate
    # NOTE: I couldn't quite get these to work, but I think it's
    # because there are A LOT of PCs... which probably means that PCA is not
    # the correct tool for this job! Or, I was using it too naively!

    # fig, ax = plt.subplots(figsize=(8, 8))
    # pcas = (1, 2)
    # arrow_scale = 3
    # ax.scatter(
    #     temporal_pc[:, pcas[0] - 1], temporal_pc[:, pcas[1] - 1], alpha=0.7
    # )
    # # for i in range(loadings.shape[0]):
    # for i in range(10):
    #     ax.arrow(
    #         0,
    #         0,
    #         loadings[i, 0] * arrow_scale,
    #         loadings[i, 1] * arrow_scale,
    #         color="red",
    #         head_width=0.05,
    #     )

    #     ax.text(
    #         loadings[i, 0] * arrow_scale * 1.1,
    #         loadings[i, 1] * arrow_scale * 1.1,
    #         f"PC{i} spatial mode",
    #         color="red",
    #     )
    # ax.set_xlabel("PC1")
    # ax.set_ylabel("PC2")
    # ax.axhline(0, color="grey", lw=0.5)
    # ax.axvline(0, color="grey", lw=0.5)
    # plt.show()
    # plt.clf()

    # Which spatial component to inspect? (zero indexed)
    component_idx = 0

    # Plot spatial EOFs
    eof = np.full(coarse_ny * coarse_nx, np.nan)
    eof[good_cells_flat] = VT[component_idx]
    eof = eof.reshape(coarse_ny, coarse_nx)
    plt.imshow(eof)
    plt.colorbar()
    plt.title(f"EOF{component_idx+1} reconstruction")
    plt.savefig(
        f"eof{component_idx+1}_reconstruction_map.png", bbox_inches="tight"
    )
    plt.clf()

    # Plot temporal spectra of the principle components
    t_years = ((tstamps - tstamps[0]) / np.timedelta64(1, "D")) / 365.25
    freq = np.linspace(1 / 30, 6, 2000)
    period = 1 / freq

    # Periodogram power at given frequencies
    power = LombScargle(t_years, temporal_pc[:, component_idx]).power(freq)
    plt.semilogx(period, power)
    plt.gca().invert_xaxis()
    plt.xlabel("Period (years)")
    plt.ylabel("Power")
    plt.title(f"Temporal spectrum of EOF{component_idx+1}")
    plt.savefig(
        f"eof{component_idx+1}_temporal_spectrum.png", bbox_inches="tight"
    )
    plt.clf()

    # Plot the spatial spectrum through time
    # Here, L is the spatial scale sampled in units of
    #   `reduction_factor * xy_scale`
    # and then we are plotting the wave number corresponding to that
    # spatial scale, k = 2*pi/L
    plt.imshow(
        spectra[:, 1:],
        interpolation="none",
        aspect="auto",
        origin="lower",
        norm="log",
    )
    plt.xlabel("Spatial modes, $k=2\pi/L$ ($m^{-1}$)")
    plt.ylabel("Time step")
    plt.colorbar(label="Power")
    plt.savefig("radial_spectra_over_time.png", bbox_inches="tight")
    plt.clf()
