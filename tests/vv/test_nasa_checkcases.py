"""Verification against NASA's 6-DOF check-cases (NASA/TM-2015-218675).

Plume (high-fidelity mode) must agree with the median of NASA's simulation tools within
max(absolute floor, 2 x the spread among the NASA tools) for every output variable.
"""

from __future__ import annotations

import pytest

from plume.analysis.nasa_checkcases import CASES, FLOORS, compare, passed

pytestmark = pytest.mark.vv


@pytest.mark.parametrize("number", sorted(CASES))
def test_nasa_atmospheric_checkcase(number):
    res = compare(number)
    failures = {
        v: (res.errors[v], max(FLOORS[v], 2 * res.spread[v])) for v in res.errors if not passed(res, v)
    }
    assert not failures, f"case {number} ({CASES[number].name}): error > tolerance for {failures}"
