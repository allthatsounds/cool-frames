"""
Phase 3 – Phase gradient and reconstruction tests.

Validates:
  - filterbankphasegrad: torch vs. NumPy agreement
  - filterbankconstphase: heap-based / fixed-order phase integration
  - constphase_nonuniform: differentiable offline phase reconstruction
  - RtConstphaseNonuniformState: streaming phase reconstruction
"""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.requires_torch_impl


def np_to_torch(x: np.ndarray, dtype=None, device=None) -> torch.Tensor:
    """Convert a NumPy array to a torch tensor, preserving complex type."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


def torch_to_np(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a NumPy array on CPU."""
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _analyse_with_phase_grad(fb, signal_np):
    """Run NumPy analysis + phase gradient as reference."""
    from cool_frames.numpy.phase import filterbankphasegrad

    tgrad, fgrad, s, c = filterbankphasegrad(
        signal_np,
        fb["g"],
        fb["a"],
        L=fb["L"],
    )
    return tgrad, fgrad, s, c


# ---------------------------------------------------------------------------
# Phase gradient computation
# ---------------------------------------------------------------------------


class TestFilterbankPhasegrad:
    """filterbankphasegrad: torch vs. NumPy."""

    def test_agreement(self, erb_filterbank, noise_signal):
        from cool_frames.torch.phase import filterbankphasegrad as torch_fn

        fb = erb_filterbank
        tgrad_np, fgrad_np, s_np, _c_np = _analyse_with_phase_grad(fb, noise_signal)

        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        tgrad_t, fgrad_t, s_t, _c_t = torch_fn(
            f_t,
            fb["g"],
            fb["a"],
            L=fb["L"],
        )

        for m in range(fb["M"]):
            np.testing.assert_allclose(
                torch_to_np(tgrad_t[m]),
                tgrad_np[m],
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"tgrad channel {m}",
            )
            np.testing.assert_allclose(
                torch_to_np(fgrad_t[m]),
                fgrad_np[m],
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"fgrad channel {m}",
            )
            np.testing.assert_allclose(
                torch_to_np(s_t[m]),
                s_np[m],
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"spectrogram channel {m}",
            )

    def test_spectrogram_non_negative(self, erb_filterbank, noise_signal):
        from cool_frames.torch.phase import filterbankphasegrad

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        _, _, s, _ = filterbankphasegrad(f_t, fb["g"], fb["a"], L=fb["L"])

        for m, sm in enumerate(s):
            assert torch.all(sm >= 0), f"spectrogram channel {m} has negatives"

    def test_tgrad_bounded(self, erb_filterbank, noise_signal):
        """tgrad (normalised inst. frequency) should be in [-2, 2]."""
        from cool_frames.torch.phase import filterbankphasegrad

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        tgrad, _, _, _ = filterbankphasegrad(f_t, fb["g"], fb["a"], L=fb["L"])

        for m in range(fb["M"]):
            tg = torch_to_np(tgrad[m])
            assert np.all(tg >= -2.0 - 1e-10), f"tgrad[{m}] below -2"
            assert np.all(tg <= 2.0 + 1e-10), f"tgrad[{m}] above 2"


# ---------------------------------------------------------------------------
# Phase reconstruction (heap-based / fixed-order PGHI)
# ---------------------------------------------------------------------------


class TestFilterbankconstphase:
    """filterbankconstphase: phase integration from magnitudes + gradients."""

    def test_agreement_with_numpy(self, erb_filterbank, noise_signal):
        from cool_frames.numpy.phase import filterbankconstphase as np_fn
        from cool_frames.torch.phase import filterbankconstphase as torch_fn

        fb = erb_filterbank
        c_np = np_fn(noise_signal, fb["g"], fb["a"], L=fb["L"])

        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c_t, _mask_t = torch_fn(f_t, fb["g"], fb["a"], L=fb["L"])

        for m in range(fb["M"]):
            # Magnitudes must match (phase only differs)
            np.testing.assert_allclose(
                np.abs(torch_to_np(c_t[m])),
                np.abs(c_np[m]),
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"magnitude mismatch channel {m}",
            )

    def test_magnitude_preservation(self, erb_filterbank, noise_signal):
        """Reconstructed coefficients preserve input magnitudes."""
        from cool_frames.torch.filterbanks import filterbank
        from cool_frames.torch.phase import filterbankconstphase

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c_orig = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

        c_recon, _ = filterbankconstphase(
            f_t,
            fb["g"],
            fb["a"],
            L=fb["L"],
        )

        for m in range(fb["M"]):
            mag_orig = torch.abs(c_orig[m])
            mag_recon = torch.abs(c_recon[m])
            np.testing.assert_allclose(
                torch_to_np(mag_recon),
                torch_to_np(mag_orig),
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"magnitude not preserved at channel {m}",
            )

    def test_output_is_complex(self, erb_filterbank, noise_signal):
        from cool_frames.torch.phase import filterbankconstphase

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c, _mask = filterbankconstphase(f_t, fb["g"], fb["a"], L=fb["L"])

        for m in range(fb["M"]):
            assert c[m].is_complex(), f"channel {m} should be complex"


# ---------------------------------------------------------------------------
# Non-uniform differentiable phase reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="constphase_nonuniform moved to audioeffects in the 2026-06 consolidation"
)
class TestConstphaseNonuniform:
    """constphase_nonuniform: offline differentiable phase integration."""

    def test_agreement_with_numpy(self, erb_filterbank, noise_signal):
        from cool_frames.numpy.filterbanks import filterbank
        from cool_frames.numpy.phase import constphase_nonuniform as np_fn
        from cool_frames.torch.phase import constphase_nonuniform as torch_fn

        fb = erb_filterbank
        c_np = filterbank(noise_signal, fb["g"], fb["a"], L=fb["L"])
        s_np = [np.abs(cm) for cm in c_np]
        a_int = fb["a"][:, 0] if fb["a"].ndim == 2 else fb["a"]
        fc_n = fb["fc"] / fb["fs"] * 2  # normalise to [0, 2]

        # Need tfr — use ones as placeholder if not available
        M = fb["M"]
        tfr = np.ones(M)

        c_list_np, _phase_np, _tgrad_np, _fgrad_np = np_fn(
            s_np,
            a_int,
            fc_n,
            tfr,
        )

        # Torch version
        s_t = [np_to_torch(sm, dtype=torch.float64) for sm in s_np]
        c_list_t, _phase_t, _tgrad_t, _fgrad_t = torch_fn(
            s_t,
            torch.from_numpy(np.asarray(a_int)),
            torch.from_numpy(fc_n),
            torch.from_numpy(tfr),
        )

        for m in range(M):
            # Phase values should be close (mod 2π ambiguity handled by magnitude check)
            mag_np = np.abs(c_list_np[m])
            mag_t = np.abs(torch_to_np(c_list_t[m]))
            np.testing.assert_allclose(
                mag_t,
                mag_np,
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"constphase_nonuniform magnitude mismatch ch {m}",
            )

    def test_output_structure(self, erb_filterbank, noise_signal):
        """Returns (c_list, phase_list, tgrad_list, fgrad_list)."""
        from cool_frames.torch.filterbanks import filterbank
        from cool_frames.torch.phase import constphase_nonuniform

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])
        s = [torch.abs(cm) for cm in c]

        a_int = fb["a"][:, 0] if fb["a"].ndim == 2 else fb["a"]
        fc_n = fb["fc"] / fb["fs"] * 2
        tfr = np.ones(fb["M"])

        result = constphase_nonuniform(
            s,
            torch.from_numpy(np.asarray(a_int)),
            torch.from_numpy(fc_n),
            torch.from_numpy(tfr),
        )

        assert len(result) == 4
        c_list, phase_list, _tgrad_list, _fgrad_list = result
        assert len(c_list) == fb["M"]
        assert len(phase_list) == fb["M"]


# ---------------------------------------------------------------------------
# Streaming phase reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="RtConstphaseNonuniformState moved to audioeffects in the 2026-06 consolidation"
)
class TestRtConstphaseNonuniformState:
    """RtConstphaseNonuniformState: streaming differentiable PGHI."""

    def test_streaming_vs_offline(self, erb_filterbank, noise_signal):
        """Streaming tick-by-tick should approximate offline result."""
        from cool_frames.torch.filterbanks import filterbank
        from cool_frames.torch.phase import (
            RtConstphaseNonuniformState,
            constphase_nonuniform,
        )

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])
        s = [torch.abs(cm) for cm in c]

        a_int = fb["a"][:, 0] if fb["a"].ndim == 2 else fb["a"]
        fc_n = fb["fc"] / fb["fs"] * 2
        tfr = np.ones(fb["M"])
        M = fb["M"]

        # Offline reference
        _c_off, _phase_off, _, _ = constphase_nonuniform(
            s,
            torch.from_numpy(np.asarray(a_int)),
            torch.from_numpy(fc_n),
            torch.from_numpy(tfr),
        )

        # Streaming: process tick by tick
        state = RtConstphaseNonuniformState(
            a=torch.from_numpy(np.asarray(a_int)),
            fc=torch.from_numpy(fc_n),
            sqtfr=torch.from_numpy(np.sqrt(tfr)),
        )

        # Find shortest channel to determine number of common ticks
        min_frames = min(s[m].shape[0] for m in range(M))
        streaming_phases = []
        for n in range(min(min_frames, 10)):
            mag_tick = torch.stack(
                [s[m][n] if n < s[m].shape[0] else torch.zeros(1) for m in range(M)]
            )
            phase_tick = state.process_tick(mag_tick)
            streaming_phases.append(phase_tick)

        # We only check that streaming produces finite output;
        # exact agreement with offline is not expected due to
        # different causal vs. non-causal gradient estimation
        for n, ph in enumerate(streaming_phases):
            assert torch.all(torch.isfinite(ph)), f"tick {n} has non-finite phases"

    def test_state_reset(self, erb_filterbank):
        """State can be created and produces output on first tick."""
        from cool_frames.torch.phase import RtConstphaseNonuniformState

        fb = erb_filterbank
        a_int = fb["a"][:, 0] if fb["a"].ndim == 2 else fb["a"]
        fc_n = fb["fc"] / fb["fs"] * 2
        tfr = np.ones(fb["M"])

        state = RtConstphaseNonuniformState(
            a=torch.from_numpy(np.asarray(a_int)),
            fc=torch.from_numpy(fc_n),
            sqtfr=torch.from_numpy(np.sqrt(tfr)),
        )

        mag = torch.ones(fb["M"], dtype=torch.float64)
        phase = state.process_tick(mag)
        assert phase.shape == (fb["M"],)
        assert torch.all(torch.isfinite(phase))
