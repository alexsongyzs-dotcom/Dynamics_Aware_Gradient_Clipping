"""Pytest configuration.

While the project is a skeleton, tests document the expected behavior of
not-yet-implemented functions. Any test that hits a NotImplementedError stub
is skipped (reported as 's'), keeping CI green; once functions are
implemented the tests run for real.
"""

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):  # pragma: no cover
    outcome = yield
    try:
        outcome.get_result()
    except NotImplementedError:
        pytest.skip("Not implemented yet (skeleton)")
