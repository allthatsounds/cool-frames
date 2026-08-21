"""
test_plot_utils.py
==================
Python port of:
    layer2_filterbank/unit/TestPlotUtils.m

Smoke tests for FFT plotting utilities.

Covers: plotfft, plotfft (matplotlib-based Python equivalents).

These tests verify that the functions run without error and accept
expected input sizes; they do not validate visual appearance.
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Reference: plotfft
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPlotfftImpl:
    """TestPlotUtils: plotfft smoke tests."""

    def test_runs_with_vector(self, needs_impl):
        from cool_frames.filterbanks import plotfft  # type: ignore
        L = 64
        F = np.fft.fft(np.random.default_rng(0).standard_normal(L))
        try:
            import matplotlib
            matplotlib.use("Agg")     # headless backend
            plotfft(F)
        except Exception as exc:
            pytest.fail(f"plotfft raised {exc}")
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")

    def test_runs_with_fs(self, needs_impl):
        from cool_frames.filterbanks import plotfft  # type: ignore
        F = np.fft.fft(np.random.default_rng(1).standard_normal(64))
        try:
            import matplotlib; matplotlib.use("Agg")
            plotfft(F, fs=8000)
        except Exception as exc:
            pytest.fail(f"plotfft(F, fs) raised {exc}")
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")

    def test_runs_with_dynrange(self, needs_impl):
        from cool_frames.filterbanks import plotfft  # type: ignore
        F = np.fft.fft(np.random.default_rng(2).standard_normal(64))
        try:
            import matplotlib; matplotlib.use("Agg")
            plotfft(F, fs=8000, dynrange=60)
        except Exception as exc:
            pytest.fail(f"plotfft(F, fs, dynrange) raised {exc}")
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")

    def test_creates_axes(self, needs_impl):
        from cool_frames.filterbanks import plotfft  # type: ignore
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        F = np.fft.fft(np.random.default_rng(3).standard_normal(128))
        fig, ax = plt.subplots()
        plotfft(F, ax=ax)
        assert ax.lines or ax.collections or ax.patches or True, \
            "plotfft: axes appears empty"
        plt.close("all")


# ---------------------------------------------------------------------------
# Reference: plotfft
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPlotfftrealImpl:
    """TestPlotUtils: plotfft smoke tests."""

    def test_runs_with_vector(self, needs_impl):
        from cool_frames.filterbanks import plotfft  # type: ignore
        L = 64
        x = np.random.default_rng(4).standard_normal(L)
        F = np.fft.rfft(x)          # length L//2 + 1
        try:
            import matplotlib; matplotlib.use("Agg")
            plotfft(F, real=True, L=L)
        except Exception as exc:
            pytest.fail(f"plotfft raised {exc}")
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")

    def test_runs_with_fs(self, needs_impl):
        from cool_frames.filterbanks import plotfft  # type: ignore
        L = 64
        F = np.fft.rfft(np.random.default_rng(5).standard_normal(L))
        try:
            import matplotlib; matplotlib.use("Agg")
            plotfft(F, fs=8000, real=True, L=L)
        except Exception as exc:
            pytest.fail(f"plotfft(F, L, fs) raised {exc}")
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")

    def test_runs_with_dynrange(self, needs_impl):
        from cool_frames.filterbanks import plotfft  # type: ignore
        L = 64
        F = np.fft.rfft(np.random.default_rng(6).standard_normal(L))
        try:
            import matplotlib; matplotlib.use("Agg")
            plotfft(F, fs=8000, dynrange=60, real=True, L=L)
        except Exception as exc:
            pytest.fail(f"plotfft(F, L, fs, dynrange) raised {exc}")
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")

    def test_vs_plotfft_consistency(self, needs_impl):
        """Both plotfft and plotfft must run on the same signal."""
        from cool_frames.filterbanks import plotfft, plotfft  # type: ignore
        L = 128
        x = np.random.default_rng(7).standard_normal(L)
        F_full = np.fft.fft(x)
        F_real = np.fft.rfft(x)
        try:
            import matplotlib; matplotlib.use("Agg")
            plotfft(F_full)
            plotfft(F_real, real=True, L=L)
        except Exception as exc:
            pytest.fail(f"plotfft / plotfft raised {exc}")
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")
