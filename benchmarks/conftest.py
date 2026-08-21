"""
pytest-benchmark configuration for cool_frames.

Baselines are stored in benchmarks/.benchmarks/ (committed to git so
regressions can be detected against any branch).

Usage
-----
Save a new baseline::

    pytest benchmarks/ --benchmark-autosave --benchmark-name=short

Compare against the most recent baseline::

    pytest benchmarks/ --benchmark-compare

Compare with strict regression threshold (5% slowdown = fail)::

    pytest benchmarks/ --benchmark-compare --benchmark-compare-fail=mean:5%
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "benchmark: mark a test as a performance benchmark",
    )
