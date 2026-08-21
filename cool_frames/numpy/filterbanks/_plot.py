"""
_plot.py
========
plotfilterbank – Python port of ``layer2/visualization/plotfilterbank.m``.

Public API
----------
    plotfilterbank(coef, a, fc=None, fs=None, dynrange=None, *,
                   xres=800, scale='db', clim=None, colorbar=True,
                   colormap=None, tc=False, ntickpos=10, tick=None,
                   audtick=False, ax=None, **kwargs)
    -> AxesImage

Parameters
----------
coef : list[ndarray] or ndarray
    Filterbank coefficients.  Either the non-uniform list-of-arrays
    returned by ``filterbank()``, or the uniform 2-D matrix returned by
    ``ufilterbank()``.  For list input each element may be 1-D (mono) or
    2-D (multi-channel); only the first channel is plotted.
a : int | array-like
    Hop size(s).  Scalar for uniform banks; 1-D or (M,2) array for
    non-uniform banks (same convention as in ``filterbank``).
fc : array-like, optional
    Centre frequencies in Hz.  If given, the y-axis is labelled with
    frequency values; otherwise channels are numbered 1 … M.
fs : float, optional
    Sample rate of the original signal in Hz.  When given the x-axis is
    labelled in seconds; otherwise in samples.
dynrange : float, optional
    Dynamic range in dB.  Only used for ``scale='db'`` and ``scale='dbsq'``.
    Clips the colour range to ``[max(C) - dynrange, max(C)]``.
xres : int, optional
    Number of display columns (time resolution of the image).
    Default 800.
scale : {'db', 'dbsq', 'linabs', 'linsq', 'lin'}, optional
    Colour-scale transformation applied to the coefficients before
    plotting.  Matches the ``tfplot`` flags in MATLAB:

    - ``'db'``     – ``20 log10(|c| + eps)``  (default)
    - ``'dbsq'``   – ``10 log10(|c|^2 + eps)``
    - ``'linabs'`` – ``|c|``
    - ``'linsq'``  – ``|c|^2``
    - ``'lin'``    – real part (requires real-valued ``coef``)

clim : (float, float), optional
    Fixed colour limits ``[cmin, cmax]``.  Overrides *dynrange*.
colorbar : bool, optional
    Whether to draw a colorbar.  Default True.
colormap : str or Colormap, optional
    Matplotlib colormap.  Default ``'viridis'``.
tc : bool, optional
    Centre the time axis around zero.  Default False.
ntickpos : int, optional
    Number of y-axis tick positions when *fc* is given and *tick* is not.
    Default 10.
tick : array-like, optional
    Explicit y-axis tick values in Hz.  Overrides *ntickpos*.
audtick : bool, optional
    Use standard auditory tick marks on the y-axis
    ``[0, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]``.
    Overrides both *ntickpos* and *tick*.
ax : matplotlib.axes.Axes, optional
    Target axes.  If ``None`` (default) the current axes are used (and
    created if necessary).

Returns
-------
im : matplotlib.image.AxesImage
    The image object returned by ``ax.imshow``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plotfilterbank"]

# Auditory tick frequencies (Hz) – matches arg_plotfilterbank.m
_AUD_TICKS = [0, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]


def plotfilterbank(
    coef,
    a,
    fc=None,
    fs=None,
    dynrange=None,
    *,
    xres: int = 800,
    scale: str = "db",
    clim=None,
    colorbar: bool = True,
    colormap=None,
    tc: bool = False,
    ntickpos: int = 10,
    tick=None,
    audtick: bool = False,
    ax=None,
):
    """Plot filterbank (or ufilterbank) coefficients as a time-frequency image.

    Parameters and return value are documented in the module docstring.
    """
    # ------------------------------------------------------------------
    # Optional matplotlib import – raise a clear message if missing
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Colormap  # noqa: F401 (type guard)
    except ImportError as exc:
        raise ImportError(
            "plotfilterbank requires matplotlib.  Install it with:\n"
            "    pip install matplotlib"
        ) from exc

    # ------------------------------------------------------------------
    # Normalise hop-size array to (M, 2) int
    # ------------------------------------------------------------------
    def _normalise_a(a_in, M):
        a_arr = np.asarray(a_in)
        if a_arr.ndim == 0:
            return np.tile([int(a_arr), 1], (M, 1))
        if a_arr.ndim == 1:
            if len(a_arr) == 1:
                return np.tile([int(a_arr[0]), 1], (M, 1))
            if len(a_arr) == M:
                return np.column_stack([a_arr.astype(int),
                                        np.ones(M, dtype=int)])
        if a_arr.ndim == 2 and a_arr.shape == (M, 2):
            return a_arr.astype(int)
        # Scalar-like fallback
        return np.tile([int(np.asarray(a_in).ravel()[0]), 1], (M, 1))

    # ------------------------------------------------------------------
    # Convert coef → (M, N) image array, compute delta_t
    # ------------------------------------------------------------------
    if isinstance(coef, (list, tuple)):
        M = len(coef)
        a2 = _normalise_a(a, M)

        # Determine L (system length)
        a_num = a2[:, 0]
        a_den = a2[:, 1]
        # N_m = number of coefficients per channel (first channel of each band)
        n_m = np.array([
            (np.asarray(c).shape[0] if np.asarray(c).ndim >= 1 else 1)
            for c in coef
        ], dtype=float)
        L_vals = n_m * a_num / a_den
        L = float(L_vals[0])   # use first channel; all should agree

        # Build (M, xres) display matrix by nearest-neighbour resampling
        coef2 = np.zeros((M, xres), dtype=complex)
        for ii, row in enumerate(coef):
            arr = np.asarray(row)
            if arr.ndim > 1:
                arr = arr[:, 0]           # take first channel for multi-channel
            arr = arr.ravel()
            n = len(arr)
            if n == 1:
                coef2[ii, :] = arr[0]
            else:
                src_idx = np.round(
                    np.linspace(0, n - 1, xres)
                ).astype(int).clip(0, n - 1)
                coef2[ii, :] = arr[src_idx]

        C = coef2
        delta_t = L / xres

    else:
        # Uniform 2-D input: shape (Nc, M) – rows=time, cols=channels
        arr = np.asarray(coef)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        Nc, M = arr.shape
        a_scalar = int(np.asarray(a).ravel()[0])

        # Resample time axis to xres columns
        src_idx = np.round(
            np.linspace(0, Nc - 1, xres)
        ).astype(int).clip(0, Nc - 1)
        C = arr[src_idx, :].T.astype(complex)  # type: ignore[assignment]   # (M, xres)
        delta_t = a_scalar * Nc / xres

    # ------------------------------------------------------------------
    # Apply colour-scale transformation
    # ------------------------------------------------------------------
    eps_val = np.finfo(float).tiny

    scale = scale.lower()
    if scale == "db":
        C = 20.0 * np.log10(np.abs(C) + eps_val)
    elif scale == "dbsq":
        C = 10.0 * np.log10(np.abs(C) ** 2 + eps_val)
    elif scale == "linabs":
        C = np.abs(C)
    elif scale == "linsq":
        C = np.abs(C) ** 2
    elif scale == "lin":
        if not np.isrealobj(C):
            raise ValueError(
                "plotfilterbank: complex-valued coef cannot be plotted with "
                "scale='lin'.  Use scale='linabs' or scale='linsq'."
            )
        C = np.real(C)
    else:
        raise ValueError(
            f"plotfilterbank: unknown scale '{scale}'.  "
            "Choose from 'db', 'dbsq', 'linabs', 'linsq', 'lin'."
        )

    C = np.real(C)   # ensure float after transform

    # ------------------------------------------------------------------
    # Dynamic-range clipping → colour limits
    # ------------------------------------------------------------------
    # clim overrides dynrange (same precedence as MATLAB)
    if clim is None and dynrange is not None:
        max_val = float(np.max(C))
        clim = (max_val - float(dynrange), max_val)

    if clim is not None:
        C = np.clip(C, clim[0], clim[1])  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Time axis
    # ------------------------------------------------------------------
    if tc:
        xr = (np.arange(xres) - xres // 2) * delta_t
    else:
        xr = np.arange(xres) * delta_t

    if fs is not None:
        xr = xr / float(fs)

    # ------------------------------------------------------------------
    # Frequency-axis tick positions
    # ------------------------------------------------------------------
    yr = np.arange(1, M + 1, dtype=float)  # channel indices 1 … M

    # ------------------------------------------------------------------
    # Actual plotting
    # ------------------------------------------------------------------
    if ax is None:
        ax = plt.gca()

    extent = [xr[0], xr[-1], yr[0] - 0.5, yr[-1] + 0.5]
    im = ax.imshow(
        C,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=colormap if colormap is not None else "viridis",
        vmin=(clim[0] if clim is not None else None),
        vmax=(clim[1] if clim is not None else None),
        interpolation="nearest",
    )

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------
    if colorbar:
        ax.figure.colorbar(im, ax=ax)

    # ------------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------------
    if fs is not None:
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xlabel("Time (samples)")

    if fc is None:
        ax.set_ylabel("Channel No.")
    else:
        fc_arr = np.asarray(fc, dtype=float)
        ax.set_ylabel("Frequency (Hz)")

        # Determine tick positions (channel indices 1 … M)
        if audtick:
            tick = _AUD_TICKS
        if tick is not None:
            tick = np.asarray(tick, dtype=float)
            # Map Hz values → channel-index positions via linear interpolation
            # on the piecewise-linear fc curve (same as MATLAB spline approach,
            # but linear is sufficient for monotone fc)
            ch_idx = np.arange(1, M + 1, dtype=float)
            valid_tick = tick[
                (tick >= fc_arr[0]) & (tick <= fc_arr[-1])
            ]
            if len(valid_tick):
                tick_pos = np.interp(valid_tick, fc_arr, ch_idx)
                ax.set_yticks(tick_pos)
                ax.set_yticklabels([str(int(t)) if t == int(t) else f"{t:.0f}"
                                    for t in valid_tick])
        else:
            # Auto: ntickpos evenly spaced channel positions → Hz labels
            n_ticks = min(ntickpos, M)
            tick_pos = np.linspace(1, M, n_ticks)
            tick_hz  = np.interp(tick_pos,
                                 np.arange(1, M + 1, dtype=float),
                                 fc_arr)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels([f"{v:.0f}" for v in tick_hz])

    return im
