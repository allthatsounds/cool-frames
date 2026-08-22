"""Phase retrieval: reconstruct audio from magnitude-only filterbank coefficients.

Supports multiple methods (PGHI, GLA, FGLA, SPSI) for recovering phase and
synthesizing audio from magnitude spectra.

Example:
    s_reconstructed, info = reconstruct(magnitude_coeff, g, a, L, Ls, method='pghi')
"""

from __future__ import annotations

import numpy as np
import torch
from cool_frames.torch.filterbanks import filterbankdual, ifilterbank
from cool_frames.torch.phase import gla, legla, spsi


def reconstruct(
    s_mag: list[torch.Tensor],
    g: list,
    a: torch.Tensor | np.ndarray,
    L: int,
    Ls: int,
    method: str = "gla",
    fc=None,
    fs: float | None = None,
    real: bool | None = None,
) -> tuple[torch.Tensor, dict]:
    """Reconstruct audio from magnitude filterbank coefficients.

    Parameters
    ----------
    s_mag : list of torch.Tensor
        Magnitude spectrogram (one tensor per channel).
    g : list
        Analysis filters (list of dicts).
    a : torch.Tensor or np.ndarray
        Hop sizes for each channel.
    L : int
        Analysis frame length (samples).
    Ls : int
        Synthesis signal length (samples).
    method : {'gla', 'fgla', 'legla', 'spsi', 'pghi'}, optional
        Phase retrieval algorithm (default ``'gla'``):

        - ``'gla'``: Griffin-Lim (iterative; the reliable default)
        - ``'fgla'``: fast Griffin-Lim with momentum (Perraudin et al.)
        - ``'legla'``: Le Roux's truncated-kernel Griffin-Lim
        - ``'spsi'``: Single-Pass Spectrogram Inversion (one pass, lower quality)
        - ``'pghi'``: phase gradient heap integration (non-iterative)

        Until v0.1.1 the GLA calls here took the library default
        ``real=False``, a convention mismatch that cost ~30 dB of magnitude
        accuracy; the mode is now derived from the filters (see ``real``).
    fc : array-like, optional
        Centre frequencies in Hz, for ``method='spsi'`` and ``method='pghi'``.
        If omitted they are recovered from the filters' transfer functions.
    fs : float, optional
        Sampling rate in Hz, required alongside an explicit ``fc``.
    real : bool, optional
        Single-sided (``True``) or two-sided (``False``) convention.  By
        default this is *derived from the filters* via
        :func:`~cool_frames.numpy.filterbanks.filterbank_is_real`, rather than
        assumed: ``True`` is right for the auditory, constant-Q and real Gabor
        designers, and wrong for complex wavelet or warped banks, where
        folding the spectrum double-counts.  Pass an explicit value to
        override the detection.

    Returns
    -------
    s_reconstructed : torch.Tensor
        Reconstructed audio signal (1D).
    info : dict
        Convergence and metadata with keys:
        - 'method': method used
        - 'n_iters': number of iterations (0 for single-pass methods)
        - 'converged': whether algorithm converged (True for single-pass)
    """
    s_mag[0].device if len(s_mag) > 0 else torch.device("cpu")

    # Convert a to numpy if needed
    if isinstance(a, torch.Tensor):
        a_np = a.cpu().numpy()
    else:
        a_np = a

    # Single-sided vs two-sided is a property of the *filters*, not something
    # to assume.  Hardcoding real=True was correct for every bank shipped by
    # the auditory/CQT designers and silently wrong for a genuinely two-sided
    # one.  Derive it, and let the caller override.
    if real is None:
        from cool_frames.numpy.filterbanks import filterbank_is_real

        real = bool(filterbank_is_real(g, a_np, L))

    if method == "pghi":
        # Phase gradient heap integration, estimating the phase gradients from
        # the magnitudes alone (Prusa/Balazs/Sondergaard).
        #
        # Until v0.1.1 this branch drew `torch.rand_like(mag)` phase and
        # reported `converged: True` — it was random phase wearing PGHI's name,
        # bitwise identical to the 'spsi' branch below, and 57 dB worse than
        # 'gla' on magnitude error.  It then briefly raised NotImplementedError,
        # because `filterbankconstphase`'s magnitude path measured no better
        # than zero phase: its gradient estimator returned the instantaneous
        # frequency *deviation* from each channel's centre frequency while the
        # heap integrator consumes the absolute value, so the centre-frequency
        # term — an order of magnitude larger than the deviation — was simply
        # missing.  With that fixed the magnitude path is 2-19x better than
        # zero phase and, on frequency-modulated signals, within a decibel of
        # what the integrator achieves from the *true* gradients.
        from cool_frames.numpy.filters._tfr import compute_tfr_from_filters
        from cool_frames.numpy.phase import filterbankconstphase

        if fc is None:
            from cool_frames.numpy.phase._centerfreq import filter_center_frequencies

            fc_use, fs_use = filter_center_frequencies(g, L), 1.0
        else:
            fc_use = np.asarray(fc.detach().cpu() if isinstance(fc, torch.Tensor) else fc)
            fs_use = float(fs) if fs is not None else 1.0

        s_np = [np.asarray(sm.detach().cpu(), dtype=float).ravel() for sm in s_mag]
        a_arr = np.atleast_1d(np.asarray(a_np))
        a_int = np.array(
            [int(a_arr[m]) if a_arr.ndim == 1 else int(a_arr[m, 0]) for m in range(len(g))],
            dtype=int,
        )
        sqtfr = np.sqrt(np.asarray(compute_tfr_from_filters(g, L), dtype=float))

        c_np, _usedmask = filterbankconstphase(s_np, a_int, fc_use, sqtfr=sqtfr, fs=fs_use)

        dev = s_mag[0].device if len(s_mag) > 0 else torch.device("cpu")
        c = [torch.as_tensor(np.asarray(cm), dtype=torch.complex128).to(dev) for cm in c_np]
        gd = filterbankdual(g, a_np, L, real=real)
        s_reconstructed = ifilterbank(c, gd, a_np, Ls, real=real)
        info = {
            "method": "pghi",
            "n_iters": 0,
            "converged": True,
        }

    elif method in ("gla", "fgla"):
        # Griffin-Lim, plain or momentum-accelerated.
        #
        # The `except Exception -> random phase` fallback that used to wrap
        # this is gone: a phase-retrieval routine that quietly degrades to
        # noise while reporting a method name is worse than one that raises.
        c, f, relres, niter = gla(s_mag, g, a_np, L=L, Ls=Ls, real=real, method=method)
        s_reconstructed = f
        info = {
            "method": method,
            "n_iters": int(niter),
            "converged": bool(len(relres) > 0 and float(relres[-1]) < 1e-6),
        }

    elif method == "legla":
        c, f, relres, niter = legla(s_mag, g, a_np, L=L, Ls=Ls, real=real)
        s_reconstructed = f
        info = {
            "method": "legla",
            "n_iters": int(niter),
            "converged": bool(len(relres) > 0 and float(relres[-1]) < 1e-6),
        }

    elif method == "spsi":
        # Single-Pass Spectrogram Inversion — genuinely single-pass, and
        # genuinely SPSI now: this used to be the same random-phase block as
        # 'pghi'.
        #
        # `spsi` needs centre frequencies. If the caller did not supply them,
        # recover them from the filters themselves (`filter_center_frequencies`
        # returns cycles/sample, hence fs=1.0).
        if fc is None:
            from cool_frames.numpy.phase._centerfreq import filter_center_frequencies

            fc_use, fs_use = filter_center_frequencies(g, L), 1.0
        else:
            fc_use = np.asarray(fc.detach().cpu() if isinstance(fc, torch.Tensor) else fc)
            fs_use = float(fs) if fs is not None else 1.0

        c, _phase = spsi(s_mag, a_np, fc_use, fs_use)
        gd = filterbankdual(g, a_np, L, real=real)
        s_reconstructed = ifilterbank(c, gd, a_np, Ls, real=real)
        info = {
            "method": "spsi",
            "n_iters": 0,
            "converged": True,
        }

    else:
        raise ValueError(
            f"method must be one of ['pghi', 'gla', 'fgla', 'legla', 'spsi'], got {method!r}"
        )

    return s_reconstructed, info
